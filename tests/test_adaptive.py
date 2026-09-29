"""Tests for adaptive frame sampling."""
import pytest
from core.sampling import adaptive_sample


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
