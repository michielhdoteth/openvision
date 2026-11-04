"""Error handling for Open Vision with recoverable error codes."""
from typing import Optional


class OpenVisionError(Exception):
    """
    Base error for all Open Vision errors.

    Attributes:
        code: Machine-readable error code (e.g., GPU_OUT_OF_MEMORY)
        message: Human-readable error description
        suggested_retry: Optional dict with suggested retry parameters
    """

    def __init__(
        self,
        code: str,
        message: str,
        suggested_retry: Optional[dict] = None,
    ):
        self.code = code
        self.message = message
        self.suggested_retry = suggested_retry
        super().__init__(f"[{code}] {message}")

    def to_dict(self) -> dict:
        result = {
            "error": self.code,
            "message": self.message,
        }
        if self.suggested_retry:
            result["suggested_retry"] = self.suggested_retry
        return result


OphanimError = OpenVisionError



class VideoNotFoundError(OpenVisionError):
    def __init__(self, path: str):
        super().__init__(
            code="VIDEO_NOT_FOUND",
            message=f"Video file not found or cannot be opened: {path}",
            suggested_retry={"check_path": True, "path": path},
        )


class UnsupportedCodecError(OpenVisionError):
    def __init__(self, path: str, codec: str = "unknown"):
        super().__init__(
            code="UNSUPPORTED_CODEC",
            message=f"Video codec '{codec}' is not supported: {path}",
            suggested_retry={
                "convert_with_ffmpeg": f"ffmpeg -i {path} -c:v libx264 output.mp4",
            },
        )


class TooManyFramesError(OpenVisionError):
    def __init__(self, requested: int, max_allowed: int = 180):
        super().__init__(
            code="TOO_MANY_FRAMES",
            message=f"Requested {requested} frames exceeds maximum of {max_allowed}. "
                    f"Reduce fps, max_frames, or use a shorter video.",
            suggested_retry={
                "max_frames": min(requested, max_allowed),
                "mode": "fast",
            },
        )


class GpuOutOfMemoryError(OpenVisionError):
    def __init__(self, message: Optional[str] = None):
        super().__init__(
            code="GPU_OUT_OF_MEMORY",
            message=message or (
                "GPU out of memory. Detailed mode exceeded available VRAM. "
                "Retry with balanced or fast mode."
            ),
            suggested_retry={
                "mode": "fast",
                "max_resolution": 512,
                "fps": 0.25,
            },
        )


class ModelNotAvailableError(OpenVisionError):
    def __init__(self, model_name: str, provider: str = "lmstudio"):
        super().__init__(
            code="MODEL_NOT_AVAILABLE",
            message=(
                f"Model '{model_name}' is not available via {provider}. "
                f"Ensure the provider is running and the model is loaded."
            ),
            suggested_retry={
                "action": "Start the provider and load the model",
                "provider": provider,
                "model": model_name,
            },
        )


class SegmentationTargetNotFoundError(OpenVisionError):
    def __init__(self, prompt: str):
        super().__init__(
            code="SEGMENTATION_TARGET_NOT_FOUND",
            message=f"Segmentation target '{prompt}' was not found in the video frames.",
            suggested_retry={
                "try_alternative_prompt": True,
                "try_dense_mode": True,
            },
        )


class LowConfidenceObservationError(OpenVisionError):
    def __init__(self, message: Optional[str] = None):
        super().__init__(
            code="LOW_CONFIDENCE_OBSERVATION",
            message=message or (
                "The observation confidence is low. "
                "Consider using a more specific question or switching to detailed mode."
            ),
            suggested_retry={
                "mode": "detailed",
                "search_mode": "dense",
            },
        )


# Mapping from exception types to error codes
ERROR_CODES = {
    FileNotFoundError: "VIDEO_NOT_FOUND",
    ValueError: "INVALID_INPUT",
    RuntimeError: "RUNTIME_ERROR",
    ImportError: "MISSING_DEPENDENCY",
    PermissionError: "PERMISSION_DENIED",
}


def format_error_for_agent(error: OpenVisionError) -> str:
    """Format error for machine-readable agent consumption."""
    import json
    return json.dumps(error.to_dict(), indent=2)


def handle_cli_error(e: Exception) -> dict:
    """Convert any exception to a standardized error dict."""
    if isinstance(e, OpenVisionError):
        return e.to_dict()

    for exc_type, code in ERROR_CODES.items():
        if isinstance(e, exc_type):
            return {
                "error": code,
                "message": str(e),
            }

    return {
        "error": "UNEXPECTED_ERROR",
        "message": str(e),
    }
