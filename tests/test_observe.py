"""Tests for observe command."""
import copy
import json

import pytest
import numpy as np
from pathlib import Path
from typer.testing import CliRunner
from cli.app import app
from openvision.models import ObserveResult, ImageResult, TimelineEntry

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
        from openvision.models import TimelineEntry
        timeline = [TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="A single event")]
        result = tl_fn(timeline)
        assert result.content == "A single event"

    def test_timeline_summary_multiple(self):
        tl_fn, _ = self._import_helpers()
        from openvision.models import TimelineEntry
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
        from openvision.models import TimelineEntry
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

    result = runner.invoke(app, ["observe", "--json", "--transcribe", "--no-diarize", "--read-text",
                                 "--segment", "--ground", "person", str(video)])
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


def _stub_video_pipeline(monkeypatch, tmp_path, duration=10.0, has_audio=True, frames=None):
    """Point observe at a fake source so no decoder, VLM, or STT model is needed."""
    import cli.commands.observe as observe
    import core.sampling

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"observe fixture")
    if frames is None:
        frames = [{"timestamp": 0.0, "timestamp_str": "00:00",
                   "image": np.zeros((12, 16, 3), dtype=np.uint8), "reason": "test"}]
    config = {
        "cache": {"directory": str(tmp_path / "runs")},
        "defaults": {"segmentation_fps": 0.25},
        "models": {"vlm": {"provider": "lmstudio", "model": "test-vlm"}},
        "gpu_policy": {"unload_after_job": True},
        "modes": {"balanced": {"fps": 0.5, "max_frames": 4, "resolution": 768}},
    }
    # A fresh copy per call: ProviderRegistry writes a resolved base_url back into
    # the vlm dict, which would otherwise leak into the next run's cache key.
    monkeypatch.setattr(observe, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(observe, "auto_downgrade_mode", lambda mode, cfg: (mode, None))
    monkeypatch.setattr(observe, "probe", lambda *_: {
        "duration_seconds": duration, "width": 16, "height": 12, "fps": 1.0,
        "codec": "h264", "frame_count": int(duration), "has_audio": has_audio,
        "has_video": True,
    })
    monkeypatch.setattr(observe, "detect_vfr", lambda *_: {"mode": "cfr"})
    monkeypatch.setattr(observe, "detect_color_range", lambda *_: {"range_type": "unknown"})
    # Long sources route through core.sampling.adaptive_sample, short ones through
    # the smart_sample imported at module scope. Both must be stubbed.
    monkeypatch.setattr(observe, "smart_sample", lambda *a, **k: list(frames))
    monkeypatch.setattr(core.sampling, "adaptive_sample", lambda *a, **k: list(frames))
    monkeypatch.setattr(observe, "_index_video_memory", lambda *a, **k: None)
    return video


def test_raw_frames_transcription_reaches_the_stt_backend(tmp_path, monkeypatch):
    """Regression: the raw-frames path raised NameError instead of calling STT.

    `_handle_video` bound neither `input_path` nor `device`, so every
    `--raw-frames --transcribe` run swallowed a NameError into a warning and
    silently returned a frame-only handoff.
    """
    import cli.commands.observe as observe

    video = _stub_video_pipeline(monkeypatch, tmp_path)

    calls = []

    def fake_transcribe(path_arg, provider, device, captions_file=None):
        calls.append((str(path_arg), provider, device))
        raise RuntimeError("parakeet model is not installed")

    monkeypatch.setattr(observe, "_transcribe_or_use_captions", fake_transcribe)

    result = runner.invoke(app, ["observe", "--raw-frames", "--transcribe",
                                 "--no-diarize", "--json", str(video)])

    assert result.exit_code == 0, result.output
    # The backend is actually reached, with the resolved source path and device.
    assert calls == [(str(video), "parakeet", "auto")]
    # The surfaced failure is the real backend error, not a NameError.
    assert "parakeet model is not installed" in result.output
    assert "input_path" not in result.output
    assert "is not defined" not in result.output
    # And it is reported as a failure rather than a silent empty transcript.
    stage = json.loads(result.stdout)["stages"]["transcription"]
    assert stage["status"] == "failed"
    assert "parakeet model is not installed" in stage["reason"]


class TestTranscriptionDefaultAndSkippedWarning:
    """Transcription is on by default; `--no-transcribe` is the opt-out.

    The default is deliberately True, so a bare `openvision observe video.mp4`
    still pays the first-run STT model download. That trade is known and
    accepted. What must not regress is that a deliberately skipped transcript is
    never presented as a complete analysis.
    """

    def test_default_transcribes(self, tmp_path, monkeypatch):
        import cli.commands.observe as observe
        from providers.parakeet import Transcript, TranscriptSegment

        video = _stub_video_pipeline(monkeypatch, tmp_path, has_audio=True)
        monkeypatch.setattr(observe, "_transcribe_or_use_captions", lambda *a, **k: Transcript(
            segments=[TranscriptSegment(0.0, 1.0, "hello", 1.0)]))
        monkeypatch.setattr(observe.ProviderRegistry, "create", lambda _cfg: None)

        result = runner.invoke(app, ["observe", "--no-diarize", "--json", str(video)])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert payload["stages"]["transcription"]["status"] == "completed"
        assert payload["audio_segments"][0]["text"] == "hello"

    def test_explicit_transcribe_flag_still_works(self, tmp_path, monkeypatch):
        import cli.commands.observe as observe
        from providers.parakeet import Transcript, TranscriptSegment

        video = _stub_video_pipeline(monkeypatch, tmp_path, has_audio=True)
        monkeypatch.setattr(observe, "_transcribe_or_use_captions", lambda *a, **k: Transcript(
            segments=[TranscriptSegment(0.0, 1.0, "flagged", 1.0)]))
        monkeypatch.setattr(observe.ProviderRegistry, "create", lambda _cfg: None)

        result = runner.invoke(app, ["observe", "-t", "--no-diarize", "--json", str(video)])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["audio_segments"][0]["text"] == "flagged"

    def test_no_transcribe_suppresses_transcription(self, tmp_path, monkeypatch):
        import cli.commands.observe as observe
        from providers.parakeet import Transcript

        video = _stub_video_pipeline(monkeypatch, tmp_path, has_audio=True)
        called = []
        monkeypatch.setattr(observe, "_transcribe_or_use_captions",
                            lambda *a, **k: called.append(1) or Transcript())
        monkeypatch.setattr(observe.ProviderRegistry, "create", lambda _cfg: None)

        result = runner.invoke(app, ["observe", "--no-diarize", "--no-transcribe", "--json",
                                     str(video)])

        assert result.exit_code == 0, result.output
        assert called == []
        payload = json.loads(result.stdout)
        assert payload["stages"]["transcription"]["status"] == "skipped"
        assert payload["audio_segments"] == []

    def test_warns_when_audio_track_is_skipped(self, tmp_path, monkeypatch):
        video = _stub_video_pipeline(monkeypatch, tmp_path, has_audio=True)

        result = runner.invoke(app, ["observe", "--no-diarize", "--no-transcribe", "--json",
                                     str(video)])

        assert result.exit_code == 0, result.output
        assert "Audio track present but transcription was skipped" in result.output
        # The note has to say how to get speech evidence back.
        assert "--transcribe" in result.output

    def test_no_warning_when_there_is_no_audio_track(self, tmp_path, monkeypatch):
        video = _stub_video_pipeline(monkeypatch, tmp_path, has_audio=False)

        result = runner.invoke(app, ["observe", "--no-diarize", "--no-transcribe", "--json",
                                     str(video)])

        assert result.exit_code == 0, result.output
        assert "transcription was skipped" not in result.output

    def test_no_warning_when_transcription_runs(self, tmp_path, monkeypatch):
        import cli.commands.observe as observe
        from providers.parakeet import Transcript, TranscriptSegment

        video = _stub_video_pipeline(monkeypatch, tmp_path, has_audio=True)
        monkeypatch.setattr(observe, "_transcribe_or_use_captions", lambda *a, **k: Transcript(
            segments=[TranscriptSegment(0.0, 1.0, "hello", 1.0)]))
        monkeypatch.setattr(observe.ProviderRegistry, "create", lambda _cfg: None)

        result = runner.invoke(app, ["observe", "--no-diarize", "--json", str(video)])

        assert result.exit_code == 0, result.output
        assert "transcription was skipped" not in result.output
        assert json.loads(result.stdout)["audio_segments"][0]["text"] == "hello"

    def test_audio_only_source_warns_when_skipped(self, tmp_path, monkeypatch):
        import cli.commands.observe as observe

        audio = tmp_path / "meeting.wav"
        audio.write_bytes(b"audio fixture")
        metadata = {"duration_seconds": 2.0, "has_audio": True, "has_video": False}
        monkeypatch.setattr(observe, "probe", lambda *_: metadata)
        monkeypatch.setattr(observe, "_handle_audio_only", lambda *a, **k: None)

        result = runner.invoke(app, ["observe", "--no-diarize", "--no-transcribe", str(audio)])

        assert result.exit_code == 0, result.output
        assert "transcription was skipped" in result.output


class TestSamplingCoverageReporting:
    """Every run must state which part of the source it actually sampled."""

    def test_coverage_record_reports_span_and_percentage(self):
        from cli.commands.observe import _sampling_coverage

        frames = [{"timestamp": t} for t in (0.0, 100.0, 200.0)]
        coverage = _sampling_coverage(frames, 400.0)

        assert coverage["frames"] == 3
        assert coverage["sampled_start_seconds"] == 0.0
        assert coverage["sampled_end_seconds"] == 200.0
        assert coverage["sampled_span_seconds"] == 200.0
        assert coverage["coverage_percent"] == 50.0
        assert coverage["duration_seconds"] == 400.0

    def test_coverage_record_handles_no_frames(self):
        from cli.commands.observe import _sampling_coverage

        coverage = _sampling_coverage([], 400.0)
        assert coverage["frames"] == 0
        assert coverage["coverage_percent"] == 0.0

    def test_human_output_reports_coverage(self, tmp_path, monkeypatch):
        video = _stub_video_pipeline(
            monkeypatch, tmp_path, duration=600.0,
            # A frame set that only reaches the first tenth of the video.
            frames=[{"timestamp": float(t), "timestamp_str": "00:00",
                     "image": np.zeros((4, 4, 3), np.uint8), "reason": "test"}
                    for t in (0.0, 30.0, 60.0)],
        )

        result = runner.invoke(app, ["observe", "--no-diarize", str(video)])

        assert result.exit_code == 0, result.output
        assert "Sampled span:" in result.output
        assert "10% coverage" in result.output
        assert "3 frames" in result.output

    def test_json_output_includes_coverage(self, tmp_path, monkeypatch):
        video = _stub_video_pipeline(monkeypatch, tmp_path, duration=600.0)

        result = runner.invoke(app, ["observe", "--no-diarize", "--json", str(video)])

        assert result.exit_code == 0, result.output
        sampling = json.loads(result.stdout)["sampling"]
        assert sampling["duration_seconds"] == 600.0
        assert sampling["coverage_percent"] == 0.0  # single frame at t=0
        assert sampling["frames"] == 1

    def test_jsonl_output_includes_coverage(self, tmp_path, monkeypatch):
        video = _stub_video_pipeline(monkeypatch, tmp_path, duration=600.0)

        result = runner.invoke(app, ["observe", "--no-diarize", "--jsonl", str(video)])

        assert result.exit_code == 0, result.output
        events = [json.loads(line) for line in result.stdout.splitlines()]
        sampling = [e for e in events if e["event"] == "sampling"]
        assert sampling, [e["event"] for e in events]
        assert sampling[0]["data"]["duration_seconds"] == 600.0
        assert "coverage_percent" in sampling[0]["data"]
        done = [e for e in events if e["event"] == "done"][0]
        assert "coverage_percent" in done["data"]

    def test_raw_frames_output_includes_coverage(self, tmp_path, monkeypatch):
        video = _stub_video_pipeline(monkeypatch, tmp_path, duration=600.0)

        result = runner.invoke(app, ["observe", "--raw-frames", "--no-diarize", "--json", str(video)])

        assert result.exit_code == 0, result.output
        assert json.loads(result.stdout)["sampling"]["duration_seconds"] == 600.0

    def test_cached_result_still_reports_coverage(self, tmp_path, monkeypatch):
        from cli.commands.observe import _display_observation

        video = _stub_video_pipeline(monkeypatch, tmp_path, duration=600.0)
        first = runner.invoke(app, ["observe", "--no-diarize", "--json", str(video)])
        assert first.exit_code == 0, first.output

        second = runner.invoke(app, ["observe", "--no-diarize", str(video)])

        assert second.exit_code == 0, second.output
        assert "Using cached result" in second.output
        assert "Sampled span:" in second.output

    def test_display_observation_prints_coverage(self, monkeypatch):
        import io

        from rich.console import Console

        import cli.commands.observe as observe

        buffer = io.StringIO()
        monkeypatch.setattr(observe, "console", Console(file=buffer, width=200, no_color=True))

        observe._display_observation({
            "summary": "A cached run.",
            "timeline": [], "entities": [], "stages": {}, "tokens": None,
            "duration_seconds": 600.0,
            "sampling": {"frames": 12, "duration_seconds": 600.0,
                         "sampled_start_seconds": 0.0, "sampled_end_seconds": 598.0,
                         "sampled_span_seconds": 598.0, "coverage_percent": 99.7},
        })

        out = buffer.getvalue()
        assert "Sampled span:" in out
        assert "100% coverage" in out


class TestSubtitleResolution:
    """`--text-anchors` must find real-world subtitle filenames.

    Resolution used to be `path.with_suffix(ext)`, so it only ever looked for
    `<stem>.vtt`. A yt-dlp download such as `workaholic.en-orig.vtt` next to
    `workaholic.mp4` was invisible and had to be renamed by hand.
    """

    VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:02.500\nFirst cue.\n\n" \
          "00:00:04.000 --> 00:00:05.000\nSecond cue.\n"
    SRT = "1\n00:00:09,000 --> 00:00:10,500\nSrt cue.\n"

    def test_finds_yt_dlp_language_and_mark_suffix(self, tmp_path):
        from cli.commands.observe import _subtitle_anchor_times

        video = tmp_path / "workaholic.mp4"
        video.write_bytes(b"stub")
        (tmp_path / "workaholic.en-orig.vtt").write_text(self.VTT, encoding="utf-8")

        assert _subtitle_anchor_times(video) == [1.0, 4.0]

    @pytest.mark.parametrize("name", [
        "clip.en.vtt", "clip.en-US.vtt", "clip.en-orig.vtt", "clip.orig.en.vtt",
        "clip.pt-BR.vtt", "clip.en.srt",
    ])
    def test_finds_suffixed_variants(self, tmp_path, name):
        from cli.commands.observe import _subtitle_anchor_times

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"stub")
        body = self.SRT if name.endswith(".srt") else self.VTT
        (tmp_path / name).write_text(body, encoding="utf-8")

        assert _subtitle_anchor_times(video), name

    def test_bare_name_still_wins_over_suffixed(self, tmp_path):
        from cli.commands.observe import _subtitle_anchor_times

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"stub")
        (tmp_path / "clip.vtt").write_text(self.VTT, encoding="utf-8")
        (tmp_path / "clip.en-orig.vtt").write_text(
            "WEBVTT\n\n00:00:07.000 --> 00:00:08.000\nOther cue.\n", encoding="utf-8")

        assert _subtitle_anchor_times(video) == [1.0, 4.0]

    def test_srt_beats_vtt(self, tmp_path):
        from cli.commands.observe import _subtitle_anchor_times

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"stub")
        (tmp_path / "clip.vtt").write_text(self.VTT, encoding="utf-8")
        (tmp_path / "clip.srt").write_text(self.SRT, encoding="utf-8")

        assert _subtitle_anchor_times(video) == [9.0]

    def test_candidate_order_is_deterministic(self, tmp_path):
        from cli.commands.observe import _subtitle_candidates

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"stub")
        (tmp_path / "clip.vtt").write_text(self.VTT, encoding="utf-8")
        (tmp_path / "clip.en.vtt").write_text(self.VTT, encoding="utf-8")
        (tmp_path / "clip.en-orig.vtt").write_text(self.VTT, encoding="utf-8")
        (tmp_path / "clip.srt").write_text(self.SRT, encoding="utf-8")

        names = [c.name for c in _subtitle_candidates(video)]

        assert names[:3] == ["clip.srt", "clip.vtt", "clip.en-orig.vtt"]
        assert set(names[3:]) == {"clip.en.vtt", "clip.ass"}
        # Stable across calls.
        assert names == [c.name for c in _subtitle_candidates(video)]

    def test_ignores_lookalike_stems(self, tmp_path):
        from cli.commands.observe import _subtitle_anchor_times

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"stub")
        (tmp_path / "clip.otherepisode.en-orig.vtt").write_text(self.VTT, encoding="utf-8")
        (tmp_path / "myclip.en-orig.vtt").write_text(self.VTT, encoding="utf-8")

        assert _subtitle_anchor_times(video) == []

    def test_missing_subtitles_returns_empty(self, tmp_path):
        from cli.commands.observe import _subtitle_anchor_times

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"stub")

        assert _subtitle_anchor_times(video) == []

    def test_text_anchors_run_reports_the_suffixed_file(self, tmp_path, monkeypatch):
        """End-to-end: --text-anchors consumes the suffixed subtitle file."""
        import cli.commands.observe as observe
        import core.video

        video = _stub_video_pipeline(monkeypatch, tmp_path)
        (tmp_path / "clip.en-orig.vtt").write_text(self.VTT, encoding="utf-8")
        captured = []

        def fake_extract_at_timestamps(path_arg, timestamps, resolution):
            captured.extend(timestamps)
            return [{"index": i, "timestamp": t, "timestamp_str": "00:00",
                     "image": np.zeros((4, 4, 3), np.uint8), "reason": "timestamp-cue"}
                    for i, t in enumerate(timestamps)]

        monkeypatch.setattr(core.video, "extract_at_timestamps", fake_extract_at_timestamps)

        result = runner.invoke(app, ["observe", "--text-anchors", "--no-diarize", str(video)])

        assert result.exit_code == 0, result.output
        assert captured == [1.0, 4.0]
        assert "added 2 frames at subtitle cues" in result.output


class TestGridSize:
    """`--grid-size` is documented but used to be absent from the CLI."""

    def test_flag_exists_and_is_documented(self):
        result = runner.invoke(app, ["observe", "--help"])
        assert result.exit_code == 0
        assert "--grid-size" in result.output

    @pytest.mark.parametrize("value", ["0", "-1", "13", "1000"])
    def test_rejects_out_of_range_with_a_clear_message(self, value):
        result = runner.invoke(app, ["observe", "--grid-size", value, "test.mp4"])

        assert result.exit_code == 2, result.output
        assert "--grid-size must be between 1 and 12" in result.output
        assert "Traceback" not in result.output

    def test_rejects_non_integer_without_a_traceback(self):
        result = runner.invoke(app, ["observe", "--grid-size", "abc", "test.mp4"])

        assert result.exit_code != 0
        assert "Traceback" not in result.output

    @pytest.mark.parametrize("size", [1, 2, 3, 4, 12])
    def test_accepted_sizes_reach_the_contact_sheet(self, tmp_path, monkeypatch, size):
        import cli.commands.observe as observe
        import core.contact_sheet as contact_sheet

        video = _stub_video_pipeline(monkeypatch, tmp_path)
        seen = {}

        def spy(frames, grid_size=(3, 3), output_path=None, **kwargs):
            seen["grid_size"] = grid_size
            seen["output_path"] = output_path
            return output_path

        monkeypatch.setattr(contact_sheet, "create_contact_sheet", spy)

        result = runner.invoke(app, ["observe", "--grid", "--grid-size", str(size),
                                     "--no-diarize", str(video)])

        assert result.exit_code == 0, result.output
        assert seen["grid_size"] == (size, size)
        assert seen["output_path"].endswith("clip_contact_sheet.jpg")
        assert f"Contact sheet ({size}x{size})" in result.output

    def test_default_stays_three_by_three(self, tmp_path, monkeypatch):
        import core.contact_sheet as contact_sheet

        video = _stub_video_pipeline(monkeypatch, tmp_path)
        seen = {}
        monkeypatch.setattr(contact_sheet, "create_contact_sheet",
                            lambda frames, grid_size=(3, 3), output_path=None, **kw:
                            seen.update(grid_size=grid_size) or output_path)

        result = runner.invoke(app, ["observe", "--grid", "--no-diarize", str(video)])

        assert result.exit_code == 0, result.output
        assert seen["grid_size"] == (3, 3)
