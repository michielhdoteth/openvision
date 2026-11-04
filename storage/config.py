"""Configuration loader for Open Vision."""
import os
import yaml
from pathlib import Path
from typing import Optional


def find_config() -> Path:
    """
    Find the config file to use.

    Explicit environment configuration wins; otherwise use the per-user file,
    then the packaged defaults.
    """
    env_config = os.environ.get("OPENVISION_CONFIG") or os.environ.get("OPHANIM_CONFIG")
    if env_config:
        env_path = Path(env_config).expanduser()
        if not env_path.is_file():
            raise FileNotFoundError(f"Config file from OPENVISION_CONFIG does not exist: {env_path}")
        return env_path

    user_config = Path.home() / ".openvision" / "config.yaml"
    if user_config.is_file():
        return user_config

    pkg_config = Path(__file__).parent.parent / "config" / "default.yaml"
    if pkg_config.exists():
        return pkg_config

    raise FileNotFoundError(
        "Cannot find config file. Set OPENVISION_CONFIG env var "
        "or ensure config/default.yaml exists."
    )


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Optional explicit path. If None, auto-discover.

    Returns:
        Dict with merged configuration values.
    """
    if config_path:
        path = Path(config_path).expanduser()
    else:
        path = find_config()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    # Auto-discovered user config is a partial override over immutable package
    # defaults. An explicitly requested file remains an exact, isolated input.
    default_path = Path(__file__).parent.parent / "config" / "default.yaml"
    user_path = Path.home() / ".openvision" / "config.yaml"
    if not config_path and path.resolve() != default_path.resolve() and default_path.is_file():
        with open(default_path, encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}
        return merge_config(defaults, config)
    return config


def get_mode_config(config: dict, mode: str) -> dict:
    """Get configuration for a specific mode (fast/balanced/detailed)."""
    modes = config.get("modes", {})
    if mode in modes:
        return modes[mode]

    defaults = config.get("defaults", {})
    return {
        "resolution": defaults.get("max_resolution", 768),
        "fps": defaults.get("fps", 0.5),
        "max_frames": defaults.get("max_frames", 60),
    }


def merge_config(base: dict, override: dict) -> dict:
    """Deep merge two config dicts. Override values win."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result
