"""Persistent local model assets owned by Talon."""

from pathlib import Path

from deepagents_talon.config import TalonConfig, TalonConfigError


def model_cache_dir(config: TalonConfig) -> Path:
    """Create the fixed model cache beneath the configured Talon root.

    Args:
        config: Runtime configuration whose parent home owns shared model assets.

    Returns:
        Canonical Hugging Face cache directory.

    Raises:
        TalonConfigError: If cache links escape containment or creation fails.
    """
    try:
        root = config.home.parent.expanduser().resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        cache = root
        for name in ("cache", "models", "huggingface"):
            cache /= name
            if cache.is_symlink():
                msg = "Talon model cache directories must not be symbolic links"
                raise TalonConfigError(msg)
            cache.mkdir(mode=0o700, exist_ok=True)
        validate_model_cache(cache)
    except (OSError, RuntimeError) as exc:
        msg = "Cannot prepare Talon model cache; check directory permissions and free disk space"
        raise TalonConfigError(msg) from exc
    return cache


def validate_model_cache(cache: Path) -> None:
    """Reject cache entries that redirect downloads outside the cache.

    Args:
        cache: Canonical model cache directory.

    Raises:
        TalonConfigError: If a descendant symbolic link escapes the cache.
    """
    for entry in cache.rglob("*"):
        if entry.is_symlink() and not entry.resolve().is_relative_to(cache):
            msg = "Talon model cache symbolic links must remain inside the model cache"
            raise TalonConfigError(msg)
