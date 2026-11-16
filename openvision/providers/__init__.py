"""Speech-to-text provider factory.

Provides a unified interface for selecting between different STT backends:
- parakeet: NVIDIA Parakeet TDT 0.6B v3 via sherpa-onnx (default)
- whisper: OpenAI Whisper via faster-whisper
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

PROVIDERS = {
    "parakeet": "openvision.providers.parakeet.ParakeetProvider",
    "whisper": "openvision.providers.whisper.WhisperProvider",
}

DEFAULT_PROVIDER = "parakeet"


def get_provider(name: str = DEFAULT_PROVIDER, config: Optional[dict] = None):
    """
    Get an STT provider instance by name.

    Args:
        name: Provider name ("parakeet" or "whisper")
        config: Provider-specific config dict. Keys:
            - device: "auto", "cpu", or "cuda"
            - model_size: (whisper only) model size, default "base"
            - compute_type: (whisper only) "float16" or "int8"

    Returns:
        Provider instance with .transcribe(), .transcribe_audio(), .try_captions()

    Raises:
        ValueError: Unknown provider name
        ImportError: Provider package not installed
    """
    if name not in PROVIDERS:
        available = ", ".join(PROVIDERS.keys())
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")

    module_path, class_name = PROVIDERS[name].rsplit(".", 1)

    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError as e:
        if name == "whisper":
            raise ImportError(
                f"Whisper provider requires 'faster-whisper'. Install with: pip install faster-whisper"
            ) from e
        elif name == "parakeet":
            raise ImportError(
                f"Parakeet provider requires 'sherpa-onnx'. Install with: pip install sherpa-onnx"
            ) from e
        raise

    provider_class = getattr(module, class_name)
    return provider_class(config or {})


def list_providers():
    """Return list of available provider names."""
    return list(PROVIDERS.keys())
