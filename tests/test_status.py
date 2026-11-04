from pathlib import Path

from cli.commands.status import _check_parakeet_model
from core import models as core_models


PARAKEET_FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]


def _no_repo_fallback(monkeypatch, tmp_path: Path) -> None:
    """Isolate tests from a repo-local dev checkout (this machine has one)."""
    monkeypatch.setattr(
        core_models,
        "repo_local_model_dir",
        lambda name: tmp_path / "absent-repo" / name,
    )


def test_parakeet_diagnostic_accepts_null_paths_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path))
    _no_repo_fallback(monkeypatch, tmp_path)

    result = _check_parakeet_model({"paths": None})

    assert result["path"] == str(tmp_path / "models" / "parakeet-tdt-0.6b-v3-int8")
    assert result["installed"] is False
    assert result["missing_files"] == PARAKEET_FILES


def test_parakeet_diagnostic_accepts_null_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path))
    _no_repo_fallback(monkeypatch, tmp_path)

    result = _check_parakeet_model({"paths": {"models_dir": None}})

    assert result["path"] == str(tmp_path / "models" / "parakeet-tdt-0.6b-v3-int8")
    assert result["installed"] is False


def test_parakeet_diagnostic_reports_installed_home_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path))
    _no_repo_fallback(monkeypatch, tmp_path)
    target = tmp_path / "models" / "parakeet-tdt-0.6b-v3-int8"
    target.mkdir(parents=True)
    for name in PARAKEET_FILES:
        (target / name).write_bytes(b"x")

    result = _check_parakeet_model({})

    assert result["installed"] is True
    assert result["origin"] == "home"
    assert result["missing_files"] == []
