"""Tests for bidirectional agent notes in video memory (no network)."""
import json
from pathlib import Path

from core.video_memory import VideoMemoryStore


def _store(tmp_path: Path) -> VideoMemoryStore:
    return VideoMemoryStore(tmp_path / "runs.db")


def _write_video(path: Path) -> Path:
    path.write_bytes(b"not really a video")
    return path


def test_note_is_recorded_and_listed(tmp_path):
    store = _store(tmp_path)
    video = _write_video(tmp_path / "clip.mp4")

    recorded = store.note(str(video), 83.0, "red dial on the left panel",
                          frame_path="frames/f_0080.jpg")
    assert recorded["at_seconds"] == 83.0
    assert recorded["notes"] == 1

    notes = store.list_notes(str(video))
    assert len(notes) == 1
    assert notes[0]["text"] == "red dial on the left panel"
    assert notes[0]["frame_path"] == "frames/f_0080.jpg"
    assert notes[0]["start_seconds"] == 83.0


def test_note_requires_existing_source(tmp_path):
    store = _store(tmp_path)
    try:
        store.note(str(tmp_path / "missing.mp4"), 1.0, "nothing")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError for a missing source file")


def _speech_segment(start, end, text, speaker=None):
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def test_build_folds_agent_notes_into_chunks(tmp_path):
    store = _store(tmp_path)
    video = _write_video(tmp_path / "clip.mp4")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    store.note(str(video), 35.0, "operator turns the red dial")
    store.build(
        str(video), str(run_dir),
        summary="Watcher observes a control panel.",
        timeline=[],
        entities=["dial"],
        transcript_segments=[_speech_segment(30.0, 40.0, "the dial", "SPEAKER_00")],
        segmentation=None,
        duration_seconds=120.0,
    )

    hits = store.search("operator", limit=5, source_path=str(video))
    assert hits, "agent note must be searchable through memory search"
    assert "[AGENT]" in hits[0]["visual_evidence"]
    assert any(e["modality"] == "agent" for e in hits[0]["evidence"])


def test_rebuild_keeps_agent_notes(tmp_path):
    store = _store(tmp_path)
    video = _write_video(tmp_path / "clip.mp4")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    store.note(str(video), 35.0, "operator turns the red dial")
    segments = [_speech_segment(30.0, 40.0, "the dial", "SPEAKER_00")]
    store.build(str(video), str(run_dir), "s", [], ["dial"], segments, None, 120.0)
    store.build(str(video), str(run_dir), "s2", [], ["dial"], segments, None, 120.0)

    assert len(store.list_notes(str(video))) == 1
    hits = store.search("operator", limit=5, source_path=str(video))
    assert hits, "note must survive a rebuild of the hierarchy"


def test_list_notes_all_when_no_source(tmp_path):
    store = _store(tmp_path)
    video = _write_video(tmp_path / "clip.mp4")
    store.note(str(video), 5.0, "a note")
    assert len(store.list_notes()) == 1


def test_memory_note_cli_writes_json(tmp_path, monkeypatch, capsys):
    """CLI surface: `openvision memory note VIDEO "text" --at 1:23 --json`."""
    from typer.testing import CliRunner

    from cli.commands import memory as memory_cmd

    video = _write_video(tmp_path / "clip.mp4")
    (tmp_path / "frames").mkdir()
    (tmp_path / "frames" / "f.jpg").write_bytes(b"img")

    import storage.config as storage_config
    import storage.paths as storage_paths

    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path))
    monkeypatch.setattr(storage_config, "load_config", lambda: {})
    monkeypatch.setattr(
        memory_cmd, "_store",
        lambda: VideoMemoryStore(tmp_path / "runs" / "runs.db"),
    )
    result = CliRunner().invoke(
        memory_cmd.app,
        ["note", str(video), "red dial on the left panel", "--at", "1:23", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["at_seconds"] == 83.0
    assert payload["note"] == "red dial on the left panel"
    assert VideoMemoryStore(tmp_path / "runs" / "runs.db").list_notes(str(video))
