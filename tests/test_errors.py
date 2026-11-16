"""Tests for error handling system."""
import pytest
from openvision.core.errors import (
    OpenVisionError,
    VideoNotFoundError,
    UnsupportedCodecError,
    TooManyFramesError,
    GpuOutOfMemoryError,
    ModelNotAvailableError,
    SegmentationTargetNotFoundError,
    LowConfidenceObservationError,
    handle_cli_error,
    format_error_for_agent,
)


class TestOpenVisionError:
    def test_base_error(self):
        err = OpenVisionError("TEST_CODE", "Test message", {"retry": True})
        assert err.code == "TEST_CODE"
        assert err.message == "Test message"
        assert err.suggested_retry == {"retry": True}

    def test_to_dict(self):
        err = OpenVisionError("TEST", "msg", {"mode": "fast"})
        d = err.to_dict()
        assert d["error"] == "TEST"
        assert d["message"] == "msg"
        assert d["suggested_retry"] == {"mode": "fast"}

    def test_to_dict_no_retry(self):
        err = OpenVisionError("TEST", "msg")
        d = err.to_dict()
        assert "suggested_retry" not in d or d["suggested_retry"] is None

    def test_string_representation(self):
        err = OpenVisionError("CODE", "description")
        assert "[CODE] description" in str(err)


class TestSpecificErrors:
    def test_video_not_found(self):
        err = VideoNotFoundError("/nonexistent/video.mp4")
        assert err.code == "VIDEO_NOT_FOUND"
        assert "nonexistent" in err.message
        assert err.suggested_retry is not None

    def test_unsupported_codec(self):
        err = UnsupportedCodecError("test.avi", "vp9")
        assert err.code == "UNSUPPORTED_CODEC"
        assert "vp9" in err.message

    def test_too_many_frames(self):
        err = TooManyFramesError(500, 180)
        assert err.code == "TOO_MANY_FRAMES"
        assert "500" in err.message
        assert err.suggested_retry["max_frames"] == 180

    def test_gpu_oom(self):
        err = GpuOutOfMemoryError()
        assert err.code == "GPU_OUT_OF_MEMORY"
        assert err.suggested_retry["mode"] == "fast"

    def test_gpu_oom_custom(self):
        err = GpuOutOfMemoryError("Custom OOM message")
        assert "Custom OOM message" in err.message

    def test_model_not_available(self):
        err = ModelNotAvailableError("gemma-4", "lmstudio")
        assert err.code == "MODEL_NOT_AVAILABLE"
        assert "gemma-4" in err.message

    def test_segmentation_not_found(self):
        err = SegmentationTargetNotFoundError("unicorn")
        assert err.code == "SEGMENTATION_TARGET_NOT_FOUND"
        assert "unicorn" in err.message

    def test_low_confidence(self):
        err = LowConfidenceObservationError()
        assert err.code == "LOW_CONFIDENCE_OBSERVATION"


class TestFormatErrorForAgent:
    def test_format_openvision_error(self):
        err = VideoNotFoundError("test.mp4")
        formatted = format_error_for_agent(err)
        assert '"error": "VIDEO_NOT_FOUND"' in formatted
        assert '"suggested_retry"' in formatted


class TestHandleCliError:
    def test_openvision_error_passthrough(self):
        err = GpuOutOfMemoryError()
        result = handle_cli_error(err)
        assert result["error"] == "GPU_OUT_OF_MEMORY"

    def test_file_not_found_mapped(self):
        try:
            open("/nonexistent/file.mp4")
        except FileNotFoundError as e:
            result = handle_cli_error(e)
            assert result["error"] == "VIDEO_NOT_FOUND"

    def test_import_error_mapped(self):
        try:
            import nonexistent_module_xyz  # noqa: F401
        except ImportError as e:
            result = handle_cli_error(e)
            assert result["error"] == "MISSING_DEPENDENCY"

    def test_unexpected_error_fallback(self):
        result = handle_cli_error(ValueError("bad value"))
        assert result["error"] == "INVALID_INPUT"
