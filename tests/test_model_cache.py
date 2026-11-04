"""Tests for the shared model-cache resolution chain (no network)."""
from pathlib import Path

import pytest

from core import models as core_models


PARAKEET_FILES = ["encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"]
WHISPER_FILES = ["model.bin", "config.json", "vocabulary.txt", "tokenizer.json"]


@pytest.fixture
def home(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path))
    monkeypatch.setattr(
        core_models, "repo_local_model_dir",
        lambda name: tmp_path / "absent-repo" / name,
    )
    return tmp_path


def _make(base: Path, name: str, files: list[str]) -> Path:
    target = base / name
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        (target / f).write_bytes(b"x")
    return target


def test_missing_resolves_to_home_install_target(home):
    path, origin = core_models.resolve_model_dir("parakeet", "parakeet-tdt-0.6b-v3-int8")
    assert origin == "missing"
    assert path == home / "models" / "parakeet-tdt-0.6b-v3-int8"


def test_home_cache_found(home):
    target = _make(home / "models", "parakeet-tdt-0.6b-v3-int8", PARAKEET_FILES)
    path, origin = core_models.resolve_model_dir("parakeet", "parakeet-tdt-0.6b-v3-int8")
    assert origin == "home"
    assert path == target
    assert core_models.check_model_status("parakeet", "parakeet-tdt-0.6b-v3-int8").installed is True


def test_config_override_wins(home):
    override = _make(home / "custom", "parakeet-tdt-0.6b-v3-int8", PARAKEET_FILES)
    _make(home / "models", "parakeet-tdt-0.6b-v3-int8", PARAKEET_FILES)
    path, origin = core_models.resolve_model_dir(
        "parakeet", "parakeet-tdt-0.6b-v3-int8",
        {"models": {"parakeet": {"model_dir": str(override)}}},
    )
    assert origin == "override"
    assert path == override


def test_override_without_files_falls_through(home):
    empty = home / "custom-empty"
    empty.mkdir(parents=True)
    path, origin = core_models.resolve_model_dir(
        "parakeet", "parakeet-tdt-0.6b-v3-int8",
        {"models": {"parakeet": {"model_dir": str(empty)}}},
    )
    assert origin == "missing"


def test_whisper_new_layout_counts_as_installed(home):
    """faster-whisper writes vocabulary.txt+tokenizer.json — accept both."""
    _make(home / "models", "whisper-base", WHISPER_FILES)
    status = core_models.check_model_status("whisper", "base")
    assert status.installed is True


def test_whisper_legacy_layout_counts_as_installed(home):
    _make(home / "models", "whisper-base",
          ["model.bin", "config.json", "vocabulary.json"])
    assert core_models.check_model_status("whisper", "base").installed is True


def test_whisper_incomplete_not_installed(home):
    _make(home / "models", "whisper-base", ["model.bin", "config.json"])
    assert core_models.check_model_status("whisper", "base").installed is False


def test_adopt_repo_model_copies_into_home(home, monkeypatch):
    monkeypatch.setattr(
        core_models, "repo_local_model_dir",
        lambda name: home / "repo" / name,
    )
    repo_dir = _make(home / "repo", "parakeet-tdt-0.6b-v3-int8", PARAKEET_FILES)
    (repo_dir / "test_wavs").mkdir()
    (repo_dir / "test_wavs" / "en.wav").write_bytes(b"audio")
    adopted = core_models.adopt_repo_model("parakeet", "parakeet-tdt-0.6b-v3-int8")
    assert adopted is not None
    assert adopted == home / "models" / "parakeet-tdt-0.6b-v3-int8"
    # only model files are adopted, never the test corpus
    assert not (adopted / "test_wavs").exists()
    assert core_models.resolve_model_dir("parakeet", "parakeet-tdt-0.6b-v3-int8")[1] == "home"


def test_adopt_returns_nothing_when_already_in_home(home):
    _make(home / "models", "parakeet-tdt-0.6b-v3-int8", PARAKEET_FILES)
    assert core_models.adopt_repo_model("parakeet", "parakeet-tdt-0.6b-v3-int8") is None


def test_cleanup_removes_partial_downloads(home):
    junk = home / "models"
    junk.mkdir(parents=True)
    (junk / "parakeet-download.tar.bz2").write_bytes(b"x" * 10)
    (junk / "model.part").write_bytes(b"x")
    removed = core_models.cleanup_partial_downloads()
    assert len(removed) == 2
    assert not (junk / "parakeet-download.tar.bz2").exists()
    assert not (junk / "model.part").exists()


def test_download_file_atomic_and_cleans_part_on_failure(home, monkeypatch):
    target = home / "models" / "whisper-base" / "model.bin"

    def boom(url, filename, reporthook=None):
        Path(filename).write_bytes(b"x" * 5)
        raise OSError("connection reset")

    monkeypatch.setattr(core_models.urllib.request, "urlretrieve", boom)
    with pytest.raises(OSError):
        core_models.download_file("https://example.com/model.bin", target)
    assert not target.exists()
    assert list(target.parent.glob("*.part")) == []


def test_download_file_success_renames(home, monkeypatch):
    target = home / "models" / "whisper-base" / "model.bin"

    def ok(url, filename, reporthook=None):
        Path(filename).write_bytes(b"payload")

    monkeypatch.setattr(core_models.urllib.request, "urlretrieve", ok)
    core_models.download_file("https://example.com/model.bin", target)
    assert target.read_bytes() == b"payload"
    assert list(target.parent.glob("*.part")) == []
