"""Tests for observe command."""
import pytest
import numpy as np
from pathlib import Path
from typer.testing import CliRunner
from cli.app import app
from models import ObserveResult, ImageResult, TimelineEntry

runner = CliRunner()


class TestObserveModels:
    def test_observe_result_creation(self):
        result = ObserveResult(
            summary="A video of a cat",
            timeline=[
                TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="Cat appears"),
            ],
            entities=["cat"],
            artifacts_dir="/tmp/runs/test",
            confidence="medium",
        )
        assert result.summary == "A video of a cat"
        assert len(result.timeline) == 1

    def test_image_result_creation(self):
        result = ImageResult(
            description="A red car on a road",
            objects=["car", "road"],
            text_detected=[],
            confidence="high",
        )
        assert result.objects == ["car", "road"]

    def test_timeline_entry_creation(self):
        entry = TimelineEntry(
            time_seconds=5.5,
            timestamp="00:05",
            observation="A person walks into frame",
            frame_path="/tmp/frames/frame_0000.jpg",
        )
        assert entry.time_seconds == 5.5
        assert entry.observation == "A person walks into frame"

    def test_observe_result_with_timeline_entries(self):
        entries = [
            TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="Start"),
            TimelineEntry(time_seconds=5.0, timestamp="00:05", observation="Middle"),
            TimelineEntry(time_seconds=10.0, timestamp="00:10", observation="End"),
        ]
        result = ObserveResult(
            summary="Test video",
            timeline=entries,
            entities=["test", "video"],
            artifacts_dir="/tmp/runs/test",
            confidence="low",
        )
        assert len(result.timeline) == 3
        assert result.timeline[1].observation == "Middle"
        assert result.confidence == "low"

    def test_image_result_defaults(self):
        result = ImageResult(
            description="A default image",
            confidence="medium",
        )
        assert result.objects == []
        assert result.text_detected == []
        assert result.confidence == "medium"

    def test_observe_result_empty_entities(self):
        result = ObserveResult(
            summary="Empty entities test",
            timeline=[],
            entities=[],
            artifacts_dir="/tmp/runs/test",
            confidence="high",
        )
        assert result.entities == []


class TestObserveCli:
    def test_observe_nonexistent(self):
        result = runner.invoke(app, ["observe", "nonexistent.mp4"])
        assert result.exit_code != 0

    def test_observe_nonexistent_image(self):
        result = runner.invoke(app, ["observe", "nonexistent.jpg"])
        assert result.exit_code != 0

    def test_observe_help_shows(self):
        result = runner.invoke(app, ["observe", "--help"])
        assert result.exit_code == 0
        assert "Analyze" in result.output or "observe" in result.output.lower()

    def test_observe_invalid_mode(self):
        result = runner.invoke(app, ["observe", "--mode", "invalid", "test.mp4"])
        # Should fail because file doesn't exist first
        assert result.exit_code != 0

    def test_observe_image_missing_lmstudio(self):
        """If no test image exists, this should still fail cleanly."""
        test_images = list(Path("test_data").glob("*.jpg")) + list(Path("test_data").glob("*.png"))
        test_images += list(Path(".").glob("*.jpg"))
        if not test_images:
            pytest.skip("No test image available")

        result = runner.invoke(app, ["observe", str(test_images[0])])
        # Either works (LM Studio running) or fails with LM Studio error
        assert result.exit_code in (0, 1)

    def test_observe_with_question_flag(self):
        result = runner.invoke(app, ["observe", "--question", "What is this?", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_json_flag(self):
        result = runner.invoke(app, ["observe", "--json", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_force_flag(self):
        result = runner.invoke(app, ["observe", "--force", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_save_observations(self):
        result = runner.invoke(app, ["observe", "--save-observations", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_fps(self):
        result = runner.invoke(app, ["observe", "--fps", "1.0", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_max_frames(self):
        result = runner.invoke(app, ["observe", "--max-frames", "10", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_all_options(self):
        result = runner.invoke(app, [
            "observe", "--mode", "fast", "--fps", "0.5", "--max-frames", "30",
            "--question", "What color?", "--json", "--force", "--save-observations",
            "test.mp4"
        ])
        assert result.exit_code != 0  # File not found


class TestBuildSummary:
    """Test the summary builder functions directly via observe module."""

    def _import_helpers(self):
        from cli.commands.observe import (
            _build_timeline_summary_fallback, _build_qa_summary
        )
        return _build_timeline_summary_fallback, _build_qa_summary

    def test_timeline_summary_single(self):
        tl_fn, _ = self._import_helpers()
        from models import TimelineEntry
        timeline = [TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="A single event")]
        result = tl_fn(timeline)
        assert result.content == "A single event"

    def test_timeline_summary_multiple(self):
        tl_fn, _ = self._import_helpers()
        from models import TimelineEntry
        timeline = [
            TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="First event"),
            TimelineEntry(time_seconds=5.0, timestamp="00:05", observation="Second event"),
            TimelineEntry(time_seconds=10.0, timestamp="00:10", observation="Last event"),
        ]
        result = tl_fn(timeline)
        assert "First event" in result.content
        assert "Last event" in result.content

    def test_timeline_summary_empty(self):
        tl_fn, _ = self._import_helpers()
        result = tl_fn([])
        assert "No observations" in result.content

    def test_qa_summary_with_answers(self):
        _, qa_fn = self._import_helpers()
        from models import TimelineEntry
        timeline = [
            TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="Short"),
            TimelineEntry(time_seconds=5.0, timestamp="00:05", observation="A much longer detailed observation about the scene"),
        ]
        result = qa_fn(timeline, "What is happening?")
        assert "Based on video analysis" in result
        assert "longer detailed" in result

    def test_qa_summary_empty(self):
        _, qa_fn = self._import_helpers()
        result = qa_fn([], "What?")
        assert "No frames" in result


def test_transcript_from_downloaded_srt_captions(tmp_path):
    from cli.commands.observe import _transcript_from_captions

    path = tmp_path / "captions.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,500\nA person speaks.\n", encoding="utf-8")
    transcript = _transcript_from_captions(str(path))
    assert transcript is not None
    assert transcript.segments[0].text == "A person speaks."
    assert transcript.segments[0].start == 1.0
    assert transcript.duration_seconds == 2.5


def test_vlm_visible_text_prompt_and_parser():
    from cli.commands.observe import _visual_prompt, _visible_text_lines

    prompt = _visual_prompt(None, True, "Describe the frame.")
    assert "Visible text:" in prompt
    assert "Do not guess" in prompt
    assert _visible_text_lines("Action: a desk\nVisible text:\n- OPEN VISION\n- 42") == ["OPEN VISION", "42"]
    assert _visible_text_lines("Visible text: none visible") == []


def test_observe_runs_vlm_audio_diarization_segmentation_grounding_and_memory_together(
    tmp_path, monkeypatch
):
    import json
    from types import SimpleNamespace
    import numpy as np
    import cli.commands.observe as observe
    import providers.locate_anything as locate_module
    from providers.base import TokenUsage, VlmResponse
    from providers.parakeet import Transcript, TranscriptSegment

    video = tmp_path / "pipeline.mp4"
    video.write_bytes(b"integration fixture")
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    config = {
        "cache": {"directory": str(tmp_path / "runs")},
        "defaults": {"segmentation_fps": 0.25},
        "models": {"vlm": {"provider": "lmstudio", "model": "test-vlm"},
                   "segmentation": {"provider": "sam3", "model": "sam3"},
                   "locate": {"base_url": "http://localhost:8000"}},
        "gpu_policy": {"unload_after_job": True},
        "modes": {"balanced": {"fps": 0.5, "max_frames": 6, "resolution": 768,
                                "segmentation": "on_demand"}},
    }
    monkeypatch.setattr(observe, "load_config", lambda: config)
    monkeypatch.setattr(observe, "auto_downgrade_mode", lambda mode, cfg: (mode, None))
    monkeypatch.setattr(observe, "probe", lambda *_: {
        "duration_seconds": 10.0, "width": 16, "height": 12, "fps": 1.0,
        "codec": "h264", "frame_count": 10, "has_audio": True, "has_video": True,
    })
    monkeypatch.setattr(observe, "detect_vfr", lambda *_: {"mode": "cfr"})
    monkeypatch.setattr(observe, "detect_color_range", lambda *_: {"range_type": "unknown"})
    monkeypatch.setattr(observe, "smart_sample", lambda *_, **__: [{
        "timestamp": 0.0, "timestamp_str": "00:00", "image": image, "reason": "test",
    }])

    prompts = []
    class FakeVlm:
        def check_health(self):
            return True
        def describe_image(self, frame, prompt):
            prompts.append(prompt)
            return VlmResponse("A person lifts a blue cup.", TokenUsage())
        def query_text(self, prompt):
            return VlmResponse("cup, person" if "extract" in prompt.lower() else "A person demonstrates a blue cup.", TokenUsage())
        def close(self):
            pass
    monkeypatch.setattr(observe.ProviderRegistry, "create", lambda _cfg: FakeVlm())

    transcript = Transcript(segments=[TranscriptSegment(1.0, 2.0, "Here is the cup.", 0.9)])
    monkeypatch.setattr(observe, "_transcribe_or_use_captions", lambda *args: transcript)

    class FakeSegmenter:
        def segment_frames(self, frames, prompt, run_dir):
            return {"frames_processed": len(frames), "objects": [{
                "time_seconds": frames[0][0], "bbox": [1, 2, 8, 9], "score": 0.91,
                "mask_path": str(run_dir / "masks" / f"{prompt}.png"),
            }]}
        def unload(self):
            pass
    monkeypatch.setattr("providers.sam.create_segmenter", lambda _cfg: FakeSegmenter())

    class FakeLocator:
        def __init__(self, _cfg):
            pass
        def check_health(self):
            return True
        def locate_frames(self, frames, query, run_dir):
            return {"results": [{"timestamp": frames[0][0], "boxes": [{
                "label": "person", "score": 0.88, "bbox": [1, 1, 10, 11],
            }]}]}
        def unload(self):
            pass
    monkeypatch.setattr(locate_module, "LocateAnythingProvider", FakeLocator)

    result = runner.invoke(app, ["observe", "--json", "--no-diarize", "--read-text", "--segment",
                                 "--ground", "person", str(video)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert "Visible text:" in prompts[0]
    assert {entry["modality"] for entry in payload["timeline"]} >= {
        "visual", "audio", "segmentation", "grounding",
    }
    assert payload["stages"]["diarization"]["status"] == "skipped"
    assert payload["audio_segments"][0]["speaker"] == ""
    assert payload["segmentation"]["status"] == "completed"
    assert payload["stages"]["grounding"]["status"] == "completed"
    assert payload["duration_seconds"] == 10.0
    from core.video_memory import VideoMemoryStore
    status = VideoMemoryStore(tmp_path / "runs" / "runs.db").status()
    assert status["chunks"] >= 1


def test_audio_only_observe_returns_transcript_as_structured_result(tmp_path, monkeypatch, capsys):
    import json
    import cli.commands.observe as observe
    from providers.parakeet import Transcript, TranscriptSegment
    from storage.cache import RunCache

    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio fixture")
    cache = RunCache(str(tmp_path / "runs"))
    transcript = Transcript(segments=[TranscriptSegment(0.5, 1.5, "Welcome.", 0.95)])
    monkeypatch.setattr(observe, "_transcribe_or_use_captions", lambda *args: transcript)
    monkeypatch.setattr(observe, "_index_video_memory", lambda *args, **kwargs: None)

    observe._handle_audio_only(
        audio, {"duration_seconds": 2.0, "has_audio": True, "has_video": False},
        cache, "audio-only-test", True, False, "parakeet", "cpu", None,
        True, False, None, False,
    )
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["audio_segments"][0]["text"] == "Welcome."
    assert payload["stages"]["visual"]["status"] == "skipped"
    assert payload["stages"]["diarization"]["status"] == "skipped"


def test_observe_cli_exposes_agent_and_pipeline_controls():
    result = runner.invoke(app, ["observe", "--help"])
    assert result.exit_code == 0
    for option in ("--jsonl", "--no-transcribe", "--no-diarize", "--read-text", "--segment", "--ground"):
        assert option in result.output
