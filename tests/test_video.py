"""Tests for core video and image modules."""
import pytest
import numpy as np
from pathlib import Path
from core.video import (
    probe,
    _format_timestamp,
    _downscale,
    estimate_processing_cost,
    auto_fps_focus,
    extract_at_timestamps,
    detect_vfr,
    detect_color_range,
)


def test_ffprobe_detects_audio_and_video_streams(monkeypatch):
    import subprocess
    from core.video import _probe_ffprobe

    class Result:
        returncode = 0
        stdout = '{"streams":[{"codec_type":"video","pix_fmt":"yuv420p","color_range":"tv"},{"codec_type":"audio"}],"format":{"bit_rate":"12345"}}'

    def run(args, **kwargs):
        assert "stream=codec_type" in args[args.index("-show_entries") + 1]
        assert "format=bit_rate" in args[args.index("-show_entries") + 1]
        return Result()
    monkeypatch.setattr(subprocess, "run", run)
    metadata = _probe_ffprobe("sample.mp4")
    assert metadata["has_audio"] is True
    assert metadata["pixel_format"] == "yuv420p"
    assert metadata["bit_rate"] == 12345


def test_probe_accepts_audio_only_media_when_opencv_cannot_open(monkeypatch):
    import core.video as video

    class ClosedCapture:
        def isOpened(self):
            return False

    monkeypatch.setattr(video.cv2, "VideoCapture", lambda *_: ClosedCapture())
    monkeypatch.setattr(video, "_probe_ffprobe", lambda *_: {
        "duration_seconds": 12.5, "has_audio": True, "has_video": False,
        "width": 0, "height": 0, "fps": 0, "frame_count": 0,
        "codec": "aac", "bit_rate": 64000, "pixel_format": "unknown",
        "vfr_mode": "unknown", "color_range": "unknown",
    })
    metadata = video.probe("audio-only.m4a")
    assert metadata["duration_seconds"] == 12.5
    assert metadata["has_audio"] is True
    assert metadata["has_video"] is False
    assert metadata["frame_count"] == 0
from core.image import (
    downscale,
    encode_base64,
    make_thumbnail,
    load_image,
)


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0) == "00:00"

    def test_minutes(self):
        assert _format_timestamp(125) == "02:05"

    def test_hours_not_expected(self):
        # Only MM:SS for now
        result = _format_timestamp(3661)
        assert result == "61:01"  # 61 minutes, 1 second


class TestDownscale:
    def test_no_resize_needed(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _downscale(img, 768)
        assert result.shape == (100, 200, 3)

    def test_downscales_width(self):
        img = np.zeros((100, 1600, 3), dtype=np.uint8)
        result = _downscale(img, 768)
        assert result.shape[1] <= 768  # width shrunk
        assert result.shape[0] < 100   # height shrunk proportionally

    def test_downscales_height(self):
        img = np.zeros((1600, 100, 3), dtype=np.uint8)
        result = _downscale(img, 768)
        assert result.shape[0] <= 768
        assert result.shape[1] < 100


class TestEncodeBase64:
    def test_encodes_jpg(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = encode_base64(img, ".jpg")
        assert result.startswith("data:image/jpeg;base64,")
        assert len(result) > 50

    def test_encodes_png(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        result = encode_base64(img, ".png")
        assert result.startswith("data:image/png;base64,")


class TestMakeThumbnail:
    def test_thumbnail_smaller(self):
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        thumb = make_thumbnail(img, 256)
        assert max(thumb.shape[:2]) <= 256


class TestEstimateProcessingCost:
    def test_low_cost(self):
        meta = {"duration_seconds": 30, "width": 640, "height": 480}
        assert estimate_processing_cost(meta, "balanced") == "low"

    def test_high_cost_long_video(self):
        meta = {"duration_seconds": 600, "width": 1920, "height": 1080}
        assert estimate_processing_cost(meta, "detailed") == "high"


class TestLoadImage:
    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent.jpg")


class TestProbeIntegration:
    """Integration test - needs a real video file."""

    def test_probe_real_video(self):
        """If there's a test video, probe it. Otherwise skip."""
        import os
        test_video = "test_data/sample.mp4"
        if not os.path.exists(test_video) and not os.path.exists("test_data"):
            pytest.skip("No test video available")

        # Use the first .mp4 found in test_data
        test_data = Path("test_data")
        videos = list(test_data.glob("*.mp4")) + list(test_data.glob("*.avi"))
        if not videos:
            pytest.skip("No test video available")

        meta = probe(str(videos[0]))
        assert meta["duration_seconds"] > 0
        assert meta["width"] > 0
        assert meta["height"] > 0
        assert meta["fps"] > 0


class TestAutoFpsFocus:
    def test_5s_returns_30(self):
        assert auto_fps_focus(5) == 30

    def test_1s_returns_30(self):
        assert auto_fps_focus(1) == 30

    def test_0_returns_30(self):
        assert auto_fps_focus(0) == 30

    def test_10s_returns_60(self):
        assert auto_fps_focus(10) == 60

    def test_15s_returns_60(self):
        assert auto_fps_focus(15) == 60

    def test_20s_returns_60(self):
        assert auto_fps_focus(20) == 60

    def test_30s_returns_60(self):
        assert auto_fps_focus(30) == 60

    def test_45s_returns_60(self):
        assert auto_fps_focus(45) == 60

    def test_60s_returns_60(self):
        assert auto_fps_focus(60) == 60

    def test_120s_returns_80(self):
        assert auto_fps_focus(120) == 80

    def test_large_returns_80(self):
        assert auto_fps_focus(600) == 80

    def test_negative_returns_30(self):
        assert auto_fps_focus(-1) == 30


class TestExtractAtTimestamps:
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_at_timestamps("nonexistent.mp4", [0.0, 1.0])

    def test_returns_correct_structure(self):
        """Integration test - needs a real video file."""
        import os
        test_data = Path("test_data")
        if not test_data.exists():
            pytest.skip("No test data available")
        videos = list(test_data.glob("*.mp4")) + list(test_data.glob("*.avi"))
        if not videos:
            pytest.skip("No test video available")

        frames = extract_at_timestamps(str(videos[0]), [0.0, 1.0])
        assert len(frames) > 0
        for f in frames:
            assert "index" in f
            assert "timestamp" in f
            assert "timestamp_str" in f
            assert "image" in f
            assert f["reason"] == "timestamp-cue"
            assert f["pinned"] is True

    def test_empty_timestamps(self):
        """Integration test with empty list."""
        import os
        test_data = Path("test_data")
        if not test_data.exists():
            pytest.skip("No test data available")
        videos = list(test_data.glob("*.mp4")) + list(test_data.glob("*.avi"))
        if not videos:
            pytest.skip("No test video available")

        frames = extract_at_timestamps(str(videos[0]), [])
        assert frames == []


class TestVFRDetection:
    """Tests for variable frame rate detection."""

    def test_detect_vfr_returns_dict(self):
        """detect_vfr always returns a dict with expected keys."""
        result = detect_vfr("nonexistent.mp4")
        assert isinstance(result, dict)
        assert "is_vfr" in result
        assert "mode" in result
        assert "variable_frames" in result
        assert "constant_frames" in result
        assert "vfr_ratio" in result

    def test_detect_vfr_nonexistent_file(self):
        """detect_vfr handles missing files gracefully."""
        result = detect_vfr("nonexistent_file_12345.mp4")
        assert result["mode"] == "unknown"
        assert result["is_vfr"] is False

    def test_detect_vfr_real_video(self):
        """Integration test with a real video."""
        import os
        test_data = Path("test_data")
        if not test_data.exists():
            pytest.skip("No test data available")
        videos = list(test_data.glob("*.mp4"))
        if not videos:
            pytest.skip("No test video available")

        result = detect_vfr(str(videos[0]))
        assert result["mode"] in ("cfr", "vfr", "unknown")
        assert isinstance(result["variable_frames"], int)
        assert isinstance(result["constant_frames"], int)
        assert 0.0 <= result["vfr_ratio"] <= 1.0


class TestColorRangeDetection:
    """Tests for color range detection."""

    def test_detect_color_range_returns_dict(self):
        """detect_color_range always returns a dict with expected keys."""
        result = detect_color_range("nonexistent.mp4")
        assert isinstance(result, dict)
        assert "range_type" in result
        assert "y_low" in result
        assert "y_high" in result

    def test_detect_color_range_nonexistent_file(self):
        """detect_color_range handles missing files gracefully."""
        result = detect_color_range("nonexistent_file_12345.mp4")
        assert result["range_type"] == "unknown"

    def test_detect_color_range_real_video(self):
        """Integration test with a real video."""
        import os
        test_data = Path("test_data")
        if not test_data.exists():
            pytest.skip("No test data available")
        videos = list(test_data.glob("*.mp4"))
        if not videos:
            pytest.skip("No test video available")

        result = detect_color_range(str(videos[0]))
        assert result["range_type"] in ("limited", "full", "unknown")
        assert isinstance(result["y_low"], int)
        assert isinstance(result["y_high"], int)


class TestProbeEnhanced:
    """Tests for enhanced probe with VFR and color info."""

    def test_probe_returns_vfr_fields(self):
        """probe() now includes vfr_mode and color_range."""
        import os
        test_data = Path("test_data")
        if not test_data.exists():
            pytest.skip("No test data available")
        videos = list(test_data.glob("*.mp4"))
        if not videos:
            pytest.skip("No test video available")

        meta = probe(str(videos[0]))
        assert "vfr_mode" in meta
        assert "color_range" in meta
        assert "has_audio" in meta
        assert "pixel_format" in meta
        assert "bit_rate" in meta
