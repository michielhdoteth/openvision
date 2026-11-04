"""GPU memory manager for VRAM-aware operation."""
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _torch():
    """Import torch on demand.

    Importing torch costs ~12s and every caller of this module paid it, even
    commands that only read VRAM through NVML. The torch paths below are
    fallbacks for machines without NVML.
    """
    try:
        import torch
        return torch
    except ImportError:
        return None


# Try to import pynvml for VRAM detection
pynvml: Any
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


def _init_pynvml():
    """Initialize NVML if available."""
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
    return False


def get_vram_info() -> dict:
    """
    Get current VRAM usage.

    Returns:
        dict with: total_gb, free_gb, used_gb, utilization_percent
        Returns zeros with error message if NVML unavailable.
    """
    result = {
        "total_gb": 0.0,
        "free_gb": 0.0,
        "used_gb": 0.0,
        "utilization_percent": 0.0,
    }

    if not _init_pynvml():
        torch = _torch()
        if torch is not None and torch.cuda.is_available():
            result["total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            result["free_gb"] = result["total_gb"] - torch.cuda.memory_allocated(0) / 1e9
            result["used_gb"] = result["total_gb"] - result["free_gb"]
            result["utilization_percent"] = (result["used_gb"] / result["total_gb"]) * 100 if result["total_gb"] > 0 else 0
        return result

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        result["total_gb"] = info.total / 1e9
        result["free_gb"] = info.free / 1e9
        result["used_gb"] = info.used / 1e9
        result["utilization_percent"] = (info.used / info.total) * 100 if info.total > 0 else 0
    except Exception as e:
        logger.warning(f"Failed to get VRAM info: {e}")

    return result


def get_gpu_name() -> str:
    """Get the GPU name string."""
    if not _init_pynvml():
        torch = _torch()
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).name
        return "unknown"

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        return name.decode("utf-8") if isinstance(name, bytes) else str(name)
    except Exception:
        return "unknown"


def check_safe_mode(config: dict) -> dict:
    """
    Check if system is in safe mode based on VRAM.

    Returns:
        dict with: safe_mode (bool), reason (str), free_gb (float)
    """
    vram = get_vram_info()
    gpu_policy = config.get("gpu_policy", {})
    min_free = gpu_policy.get("min_free_vram_gb", 1.5)
    total_vram = config.get("machine", {}).get("vram_gb", 6)

    result = {
        "safe_mode": True,
        "reason": "Unknown VRAM state",
        "free_gb": vram["free_gb"],
        "total_gb": vram["total_gb"] or total_vram,
    }

    if vram["total_gb"] == 0:
        result["safe_mode"] = True
        result["reason"] = "Cannot detect VRAM"
        return result

    if vram["free_gb"] >= min_free * 2:
        result["safe_mode"] = False
        result["reason"] = "Sufficient VRAM"
    elif vram["free_gb"] >= min_free:
        result["safe_mode"] = False
        result["reason"] = f"VRAM tight ({vram['free_gb']:.1f}GB free), but above minimum"
    else:
        result["safe_mode"] = True
        result["reason"] = f"Low VRAM: {vram['free_gb']:.1f}GB free (min: {min_free}GB)"

    return result


def auto_downgrade_mode(requested_mode: str, config: dict) -> tuple[str, str]:
    """
    Auto-downgrade mode based on VRAM.
    Returns (actual_mode, warning_message).
    """
    safety = check_safe_mode(config)
    fallback_resolution = config.get("gpu_policy", {}).get("fallback_resolution", 512)

    if not safety["safe_mode"]:
        return requested_mode, ""

    if requested_mode == "detailed":
        return "balanced", (
            f"Downgraded from 'detailed' to 'balanced' (low VRAM: "
            f"{safety['free_gb']:.1f}GB free). Use --mode fast if this persists."
        )
    elif requested_mode == "balanced":
        return "fast", (
            f"Downgraded from 'balanced' to 'fast' (low VRAM: "
            f"{safety['free_gb']:.1f}GB free). Available VRAM too low for standard processing."
        )
    else:
        return "fast", (
            f"Staying in 'fast' mode (low VRAM: {safety['free_gb']:.1f}GB free). "
            f"Consider closing other GPU applications."
        )


def unload_model(model_name: Optional[str] = None):
    """
    Unload models from GPU memory.
    Runs torch.cuda.empty_cache().
    If model_name is provided, log which model is being unloaded.
    """
    if model_name:
        logger.info(f"Unloading model: {model_name}")

    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"CUDA cache cleared. VRAM free: {get_vram_info()['free_gb']:.1f}GB")
    else:
        logger.info("CUDA not available, no GPU memory to clear")


def vram_context(config: dict):
    """
    Context manager: tracks VRAM before/after an operation.
    Usage: with vram_context(config): run_model()
    """
    import contextlib

    @contextlib.contextmanager
    def _vram_context():
        before = get_vram_info()
        logger.debug(f"VRAM before: {before['free_gb']:.1f}GB free")
        try:
            yield
        finally:
            after = get_vram_info()
            used_delta = before["free_gb"] - after["free_gb"]
            logger.debug(f"VRAM after: {after['free_gb']:.1f}GB free (delta: {used_delta:.1f}GB)")

            # If VRAM dropped below threshold, auto-clean
            if after["free_gb"] < config.get("gpu_policy", {}).get("min_free_vram_gb", 1.5):
                unload_model()

    return _vram_context()


def get_device() -> str:
    """Get the best available device (cuda or cpu)."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def log_vram(step: str = ""):
    """Log current VRAM state for debugging."""
    vram = get_vram_info()
    prefix = f"[{step}] " if step else ""
    if vram["total_gb"] > 0:
        logger.info(
            f"{prefix}VRAM: {vram['used_gb']:.1f}/{vram['total_gb']:.1f}GB "
            f"({vram['utilization_percent']:.0f}%) used, {vram['free_gb']:.1f}GB free"
        )
    else:
        logger.info(f"{prefix}VRAM: unable to detect")
