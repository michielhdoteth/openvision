"""Tests for adaptive frame sampling."""
import pytest
from core.sampling import adaptive_sample


def _coarse_pass(duration: float, count: int = 300):
    """Frame dicts spanning `duration`, like extract_frames() would return."""
    return [{"timestamp": i * duration / (count - 1), "image": None} for i in range(count)]


def test_adaptive_sample_spans_the_full_duration_when_scenes_cluster_at_the_head(monkeypatch):
    """Regression: a change-dense opening consumed the whole frame budget.

    `adaptive_sample` took `scene_candidates[:max_frames]`. On a 23m52s video
    with a busy opening that produced 12 frames all inside the first 162s -- 89%
    of the video unsampled, reported as a success.
    """
    import core.video

    duration, coarse_count = 1432.0, 300
    frames = _coarse_pass(duration, coarse_count)
    monkeypatch.setattr(core.video, "probe",
                        lambda path: {"duration_seconds": duration, "fps": 30.0})
    monkeypatch.setattr(core.video, "extract_frames", lambda *a, **k: frames)
    # Scene changes only in the first 5% of the video.
    head = [i for i, f in enumerate(frames) if f["timestamp"] <= duration * 0.05]
    assert head
    monkeypatch.setattr("core.sampling.detect_scenes", lambda *a, **k: head)

    result = adaptive_sample("fake.mp4", max_frames=12)

    assert len(result) == 12
    assert result[0]["timestamp"] < duration * 0.02
    assert result[-1]["timestamp"] >= duration * 0.9
    span = result[-1]["timestamp"] - result[0]["timestamp"]
    assert span >= duration * 0.85, f"only {span:.0f}s of {duration:.0f}s sampled"


def test_adaptive_sample_spans_the_full_duration_without_any_scene_changes(monkeypatch):
    import core.video

    duration, coarse_count = 1432.0, 300
    frames = _coarse_pass(duration, coarse_count)
    monkeypatch.setattr(core.video, "probe",
                        lambda path: {"duration_seconds": duration, "fps": 30.0})
    monkeypatch.setattr(core.video, "extract_frames", lambda *a, **k: frames)
    monkeypatch.setattr("core.sampling.detect_scenes", lambda *a, **k: [])

    result = adaptive_sample("fake.mp4", max_frames=12)

    assert len(result) == 12
    assert result[-1]["timestamp"] >= duration * 0.9


def test_adaptive_sample_never_exceeds_the_frame_budget(monkeypatch):
    import core.video

    duration, coarse_count = 600.0, 300
    frames = _coarse_pass(duration, coarse_count)
    monkeypatch.setattr(core.video, "probe",
                        lambda path: {"duration_seconds": duration, "fps": 30.0})
    monkeypatch.setattr(core.video, "extract_frames", lambda *a, **k: frames)
    monkeypatch.setattr("core.sampling.detect_scenes",
                        lambda *a, **k: list(range(0, coarse_count, 3)))

    for budget in (1, 2, 5, 12, 40, 60, 300, 500):
        result = adaptive_sample("fake.mp4", max_frames=budget)
        assert len(result) <= budget, f"budget {budget} produced {len(result)}"
        assert result == sorted(result, key=lambda f: f["timestamp"])


class TestAdaptiveSample:

    def test_decodes_video_only_once(self, monkeypatch):
        import core.video
        calls = []
        frames = [{"timestamp": float(i), "image": None} for i in range(10)]
        monkeypatch.setattr(core.video, "probe", lambda path: {"duration_seconds": 10, "fps": 1})
        def extract(*args, **kwargs):
            calls.append(1)
            return frames
        monkeypatch.setattr(core.video, "extract_frames", extract)
        monkeypatch.setattr("core.sampling.detect_scenes", lambda *a, **k: [3, 6])
        result = adaptive_sample("fake.mp4", max_frames=10)
        assert len(calls) == 1
        assert len(result) == 10

    def test_with_real_video(self):
        """Test with our real video file."""
        from pathlib import Path
        videos = []
        dl = Path(r"C:\Users\michi\Downloads\Videos")
        if dl.exists():
            videos = list(dl.glob("*.mp4"))
        if not videos:
            test_data = Path("test_data")
            if test_data.exists():
                videos = list(test_data.glob("*.mp4"))
        if not videos:
            pytest.skip("No test video available")

        result = adaptive_sample(str(videos[0]), max_frames=60, max_resolution=768)
        assert len(result) > 0
        assert len(result) <= 60
        print(f"Adaptive sample returned {len(result)} frames")
        for f in result:
            print(f"  {f['timestamp_str']}")

        # Verify they span the full video
        first_ts = result[0]["timestamp"]
        last_ts = result[-1]["timestamp"]
        print(f"  Coverage: {first_ts:.0f}s to {last_ts:.0f}s")
        assert last_ts > first_ts

    def test_nonexistent_video(self):
        result = adaptive_sample("nonexistent.mp4", max_frames=10)
        assert result == []

    def test_minimal_params(self):
        """Should work with default params on real video."""
        from pathlib import Path
        dl = Path(r"C:\Users\michi\Downloads\Videos")
        if not dl.exists():
            pytest.skip("No test video available")
        videos = list(dl.glob("*.mp4"))
        if not videos:
            pytest.skip("No test video available")

        result = adaptive_sample(str(videos[0]))
        assert len(result) > 0
