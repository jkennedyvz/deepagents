from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from deepagents_talon import speech
from deepagents_talon.config import TalonConfig, TalonConfigError
from deepagents_talon.interfaces import ChannelMessage
from deepagents_talon.model_cache import model_cache_dir


@pytest.fixture
def config(tmp_path):
    return TalonConfig.from_env({}, base_home=tmp_path / "talon")


@pytest.fixture
def loaders(monkeypatch):
    speech._local_pipelines.clear()

    def download(*, repo_id, cache_dir, token):
        assert token is False
        snapshot = Path(cache_dir) / repo_id.replace("/", "--") / "snapshots" / "revision"
        snapshot.mkdir(parents=True, exist_ok=True)
        if not (snapshot / "config.json").exists():
            (snapshot / "config.json").write_text("{}")
        return str(snapshot)

    def pipeline(task, *, model, device, trust_remote_code, model_kwargs):
        assert task == "automatic-speech-recognition"
        assert device == "cpu"
        assert trust_remote_code is False
        assert model_kwargs == {"local_files_only": True}
        assert (Path(model) / "config.json").is_file()
        return Mock(return_value={"text": "persisted"})

    hub = Mock(side_effect=download)
    factory = Mock(side_effect=pipeline)
    modules = {
        "huggingface_hub": SimpleNamespace(snapshot_download=hub),
        "transformers": SimpleNamespace(pipeline=factory),
    }
    monkeypatch.setattr(speech.importlib, "import_module", modules.__getitem__)
    yield hub, factory
    speech._local_pipelines.clear()


def test_cache_defaults_and_configured_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = TalonConfig.from_env({})
    assert model_cache_dir(default) == tmp_path / ".deepagents/cache/models/huggingface"
    custom = TalonConfig.from_env({"DEEPAGENTS_TALON_HOME": str(tmp_path / "custom")})
    assert model_cache_dir(custom) == tmp_path / "custom/cache/models/huggingface"


@pytest.mark.parametrize("component", ["cache", "cache/models", "cache/models/huggingface"])
def test_cache_rejects_directory_symlinks(config, tmp_path, component):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = config.home.parent / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(TalonConfigError, match="symbolic links"):
        model_cache_dir(config)
    assert list(outside.iterdir()) == []


def test_cache_rejects_escaping_descendants(config, tmp_path):
    cache = model_cache_dir(config)
    (cache / "blobs").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(TalonConfigError, match="inside the model cache"):
        model_cache_dir(config)


def test_cache_accepts_huggingface_internal_links(config):
    cache = model_cache_dir(config)
    (cache / "blob").write_text("weights")
    (cache / "snapshot").symlink_to("blob")
    assert model_cache_dir(config) == cache


def test_cache_creation_failure(config):
    config.home.parent.mkdir(parents=True)
    (config.home.parent / "cache").write_text("blocked")
    with pytest.raises(TalonConfigError, match="Cannot prepare Talon model cache"):
        model_cache_dir(config)


def test_pipeline_persistence_and_home_isolation(config, loaders, monkeypatch, tmp_path):
    hub, factory = loaders
    for name in ("HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.setenv(name, str(tmp_path / "outside"))
    first = speech._load_local_pipeline("nvidia/parakeet", "cpu", config)
    assert speech._load_local_pipeline("nvidia/parakeet", "cpu", config) is first
    snapshot = Path(factory.call_args.kwargs["model"])
    (snapshot / "config.json").write_text("persisted assets")
    speech._local_pipelines.clear()
    second = speech._load_local_pipeline("nvidia/parakeet", "cpu", config)
    assert second("audio") == {"text": "persisted"}
    assert (snapshot / "config.json").read_text() == "persisted assets"
    assert factory.call_args_list[0].kwargs["model"] == factory.call_args_list[1].kwargs["model"]
    other = TalonConfig.from_env({}, base_home=tmp_path / "other")
    speech._load_local_pipeline("nvidia/parakeet", "cpu", other)
    assert hub.call_args.kwargs["cache_dir"] == str(model_cache_dir(other))
    assert not (tmp_path / "outside").exists()


def test_concurrent_initialization(config, loaders):
    _, factory = loaders
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(speech._load_local_pipeline, "nvidia/parakeet", "cpu", config)
            for _ in range(4)
        ]
        pipelines = [future.result() for future in futures]
    assert all(pipeline is pipelines[0] for pipeline in pipelines)
    assert factory.call_count == 1


def test_failed_download_is_retryable(config, loaders):
    hub, _ = loaders
    download = hub.side_effect
    hub.side_effect = OSError("download interrupted")
    with pytest.raises(OSError, match="download interrupted"):
        speech._load_local_pipeline("nvidia/parakeet", "cpu", config)
    hub.side_effect = download
    assert speech._load_local_pipeline("nvidia/parakeet", "cpu", config)("audio") == {
        "text": "persisted"
    }


def test_snapshot_outside_cache_rejected(config, loaders, tmp_path):
    hub, factory = loaders
    hub.side_effect = None
    hub.return_value = str(tmp_path)
    with pytest.raises(ValueError, match="snapshot must remain inside"):
        speech._load_local_pipeline("nvidia/parakeet", "cpu", config)
    factory.assert_not_called()


async def test_transcription_uses_configured_cache(config, loaders, monkeypatch, tmp_path):
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    wav = tmp_path / "converted.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(speech, "_convert_to_wav", lambda _: wav)
    enabled = TalonConfig.from_env({"SPEECH_ENABLED": "true"}, base_home=config.home.parent)
    transcriber = speech.build_voice_transcriber(enabled)
    message = ChannelMessage(conversation_id="chat", text="", metadata={"voice_path": str(audio)})
    result = await speech.transcribe_voice_message(transcriber, message)
    assert result.text == "persisted"
    assert not wav.exists()
    hub, _ = loaders
    assert hub.call_args.kwargs["cache_dir"] == str(model_cache_dir(config))


async def test_cache_failure_preserves_message(config, monkeypatch, tmp_path, caplog):
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")
    wav = tmp_path / "converted.wav"
    wav.write_bytes(b"wav")
    monkeypatch.setattr(speech, "_convert_to_wav", lambda _: wav)
    config.home.parent.mkdir(parents=True)
    (config.home.parent / "cache").write_text("blocked")
    message = ChannelMessage(
        conversation_id="chat", text="original", metadata={"voice_path": str(audio)}
    )
    transcriber = speech.LocalParakeetVoiceTranscriber(config=config)
    assert await speech.transcribe_voice_message(transcriber, message) == message
    assert "Cannot prepare Talon model cache" in caplog.text
    assert not wav.exists()


def test_assistants_share_assets(config, loaders):
    other = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "other"}, base_home=config.home.parent)
    first = speech._load_local_pipeline("nvidia/parakeet", "cpu", config)
    assert speech._load_local_pipeline("nvidia/parakeet", "cpu", other) is first
    hub, _ = loaders
    assert hub.call_count == 1
