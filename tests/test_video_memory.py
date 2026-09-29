"""Tests for the hierarchical and explicitly grounded video memory."""
import json
from typer.testing import CliRunner

from cli.app import app
from storage.cache import RunCache

from core.video_memory import VideoMemoryStore


def test_build_and_search_keeps_modalities_and_identity_unresolved(tmp_path):
    video = tmp_path / "meeting.mp4"
    video.write_bytes(b"fixture")
    run = tmp_path / "run"
    run.mkdir()
    store = VideoMemoryStore(tmp_path / "runs.db")
    result = store.build(
        str(video), run, "A person presents a design review.",
        [
            {"time_seconds": 4, "timestamp": "00:04", "observation": "A presenter points at a blue chart.", "modality": "visual", "frame_path": "frame.jpg"},
            {"time_seconds": 5, "timestamp": "00:05", "observation": "We should ship the blue chart tomorrow.", "modality": "audio"},
        ],
        entities=["presenter", "blue chart"],
        transcript_segments=[
            {"start": 5, "end": 7, "speaker": "SPEAKER_00", "text": "We should ship the blue chart tomorrow."},
            {"start": 605, "end": 607, "speaker": "SPEAKER_01", "text": "The meeting is almost over."},
        ],
        duration_seconds=640,
    )
    matches = store.search("ship tomorrow", source_path=str(video))
    assert result["chunks"] == 2
    assert matches
    evidence = matches[0]
    assert "points at a blue chart" in evidence["visual_evidence"]
    assert "ship the blue chart tomorrow" in evidence["speech_evidence"]
    assert evidence["speakers"] == ["SPEAKER_00"]
    assert evidence["speaker_to_visible_person"] == "unverified"
    assert json.loads(json.dumps(evidence["evidence"]))[1]["modality"] == "audio"
    assert store.status()["root_nodes"] == 1
    assert store.status()["sections"] == 2


def test_rebuild_replaces_source_hierarchy_without_duplicates(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source")
    store = VideoMemoryStore(tmp_path / "runs.db")
    args = (str(video), tmp_path / "run", "Summary", [], [], [], None, 0)
    store.build(*args)
    store.build(*args)
    assert store.status()["sources"] == 1
    assert store.status()["root_nodes"] == 1


def test_memory_search_keeps_entity_matches_in_their_own_time_intervals(tmp_path):
    video = tmp_path / "wildlife.mp4"
    video.write_bytes(b"fixture")
    store = VideoMemoryStore(tmp_path / "runs.db")
    store.build(
        str(video), tmp_path / "run", "",
        [
            {"time_seconds": 5, "observation": "A falcon circles above the ridge."},
            {"time_seconds": 75, "observation": "An otter swims across the river."},
        ], entities=["falcon", "otter"], duration_seconds=90,
    )

    falcon_hits = store.search("falcon", source_path=str(video))
    assert len(falcon_hits) == 1
    assert falcon_hits[0]["start_seconds"] == 0


def test_sources_command_lists_indexed_video_hierarchy(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path / "openvision-home"))
    video = tmp_path / "ma\u0301quina.mp4"
    video.write_bytes(b"fixture")
    store = VideoMemoryStore(tmp_path / "openvision-home" / "runs" / "runs.db")
    store.build(str(video), tmp_path / "run", "A lecture about memory.", [], duration_seconds=12)

    result = CliRunner().invoke(app, ["memory", "sources", "--json"])

    assert result.exit_code == 0
    source = json.loads(result.stdout)[0]
    assert source["source_path"] == str(video.resolve())
    assert source["end_seconds"] == 12
    assert source["chunks"] == 0


def test_memory_terminal_text_escapes_unrepresentable_characters():
    from cli.commands.memory import _console_safe

    assert _console_safe("ma\u0301quina", "cp1252") == r"ma\u0301quina"


def test_memory_status_command_uses_configured_local_home(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("OPENVISION_HOME", str(tmp_path / "openvision-home"))
    result = CliRunner().invoke(app, ["memory", "status", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["sources"] == 0


def test_memory_uses_the_same_absolute_cache_directory_as_observe(tmp_path, monkeypatch):
    import yaml

    cache_dir = tmp_path / "shared-cache"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({"cache": {"directory": str(cache_dir)}}), encoding="utf-8")
    monkeypatch.setenv("OPENVISION_CONFIG", str(config_path))
    store = VideoMemoryStore(cache_dir / "runs.db")
    video = tmp_path / "shared.mp4"
    video.write_bytes(b"fixture")
    store.build(str(video), tmp_path / "run", "Shared database", [], duration_seconds=5)

    result = CliRunner().invoke(app, ["memory", "status", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sources"] == 1
    assert payload["database"] == str(cache_dir / "runs.db")


def test_observe_indexing_persists_searchable_transcript_and_visual(tmp_path):
    from cli.commands.observe import _index_video_memory
    from models import ObserveResult, TimelineEntry
    video = tmp_path / "source.mp4"
    video.write_bytes(b"media")
    cache = RunCache(str(tmp_path / "runs"))
    run_dir = cache.create_run("memorytest")
    result = ObserveResult(
        summary="A speaker demonstrates a device.",
        timeline=[TimelineEntry(time_seconds=12, timestamp="00:12", observation="A person turns the red dial.")],
        entities=["person", "red dial"], artifacts_dir=str(run_dir), confidence="medium",
    )
    class Segment:
        start, end, speaker, text = 13.0, 15.0, "SPEAKER_00", "Turn the red dial now."
    class Transcript:
        segments = [Segment()]

    _index_video_memory(cache, video, run_dir, result, Transcript(), duration_seconds=30)
    store = VideoMemoryStore(cache._db_path)
    matches = store.search("red dial", source_path=str(video))
    assert matches
    assert "turns the red dial" in matches[0]["visual_evidence"]
    assert "Turn the red dial now" in matches[0]["speech_evidence"]
    assert (run_dir / "video_memory.json").is_file()
    assert result.stages["memory"]["status"] == "completed"
    saved = json.loads((run_dir / "observations.json").read_text(encoding="utf-8"))
    assert saved["stages"]["memory"]["status"] == "completed"
