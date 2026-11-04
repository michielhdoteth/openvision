"""Tests for transcribe persistence into the run cache and video memory."""
import pytest
from typer.testing import CliRunner

from cli.app import app
from core.video_memory import VideoMemoryStore
from storage.cache import RunCache
from storage.paths import run_cache_dir


@pytest.fixture
def fake_provider(tmp_path, monkeypatch):
    """Patch the STT provider so no GPU/model is needed."""
    from providers.parakeet import Transcript, TranscriptSegment

    class FakeTranscript(Transcript):
        def __init__(self):
            self.segments = [
                TranscriptSegment(start=0.0, end=4.0, text="first sentence.", confidence=1.0),
                TranscriptSegment(start=4.0, end=9.0, text="second sentence.", confidence=1.0),
            ]
            self.language = "en"
            self.duration_seconds = 9.0

    class FakeProvider:
        def transcribe(self, path):
            return FakeTranscript()

    import cli.commands.transcribe as mod
    monkeypatch.setattr(mod, "get_provider", lambda name, config: FakeProvider())
    return mod


def _make_video(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-media")
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path / "home"))
    return video


def test_transcribe_persists_to_runs_and_video_memory(tmp_path, monkeypatch, fake_provider):
    video = _make_video(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, ["transcribe", str(video), "--provider", "parakeet"])
    assert result.exit_code == 0, result.output

    cache = RunCache(str(run_cache_dir(_config())))
    runs = cache.list_runs()
    assert runs, "transcribe must create a cached run"
    assert runs[0]["metadata"].get("command") == "transcribe"

    store = VideoMemoryStore(cache._db_path)
    matches = store.search("second sentence", source_path=str(video))
    assert matches, "transcript must be searchable for its own video"


def test_transcribe_no_run_skips_persistence(tmp_path, monkeypatch, fake_provider):
    video = _make_video(tmp_path, monkeypatch)

    result = CliRunner().invoke(app, ["transcribe", str(video), "--no-run"])
    assert result.exit_code == 0, result.output

    cache = RunCache(str(run_cache_dir(_config())))
    assert cache.list_runs() == []
    store = VideoMemoryStore(cache._db_path)
    assert not store.search("sentence", source_path=str(video))


def test_transcribe_save_to_writes_exact_path(tmp_path, monkeypatch, fake_provider):
    video = _make_video(tmp_path, monkeypatch)
    out = tmp_path / "nested" / "out.txt"

    result = CliRunner().invoke(app, ["transcribe", str(video), "--no-run", "--save-to", str(out)])
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "first sentence" in out.read_text(encoding="utf-8")


def _config():
    from storage.config import load_config
    try:
        return load_config()
    except FileNotFoundError:
        return {}
