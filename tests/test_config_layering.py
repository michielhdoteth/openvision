"""Configuration discovery and user override behavior."""
import pytest

from storage.config import find_config, load_config


def test_user_config_layers_over_packaged_defaults(tmp_path, monkeypatch):
    user_home = tmp_path / "home"
    config_dir = user_home / ".openvision"
    config_dir.mkdir(parents=True)
    user_cfg = config_dir / "config.yaml"
    user_cfg.write_text("models:\n  vlm:\n    provider: lmstudio\n", encoding="utf-8")
    monkeypatch.setattr("pathlib.Path.home", lambda: user_home)
    monkeypatch.delenv("OPENVISION_CONFIG", raising=False)
    monkeypatch.delenv("OPHANIM_CONFIG", raising=False)

    assert find_config() == user_cfg
    config = load_config()
    assert config["models"]["vlm"]["provider"] == "lmstudio"
    assert config["modes"]["balanced"]


def test_missing_explicit_config_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVISION_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError, match="OPENVISION_CONFIG"):
        find_config()
