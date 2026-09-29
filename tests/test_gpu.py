"""Tests for GPU memory manager."""
import pytest
from core.gpu import (
    get_vram_info,
    check_safe_mode,
    auto_downgrade_mode,
    get_device,
    get_gpu_name,
)


class TestGetVramInfo:
    def test_returns_dict(self):
        info = get_vram_info()
        assert isinstance(info, dict)
        assert "total_gb" in info
        assert "free_gb" in info
        assert "used_gb" in info
        assert isinstance(info["total_gb"], float)

    def test_numeric_values(self):
        info = get_vram_info()
        assert info["total_gb"] >= 0
        assert info["free_gb"] >= 0
        assert info["used_gb"] >= 0


class TestCheckSafeMode:
    def test_with_config(self):
        config = {
            "gpu_policy": {"min_free_vram_gb": 1.5},
            "machine": {"vram_gb": 6},
        }
        safety = check_safe_mode(config)
        assert "safe_mode" in safety
        assert "reason" in safety
        assert isinstance(safety["safe_mode"], bool)

    def test_empty_config(self):
        safety = check_safe_mode({})
        assert isinstance(safety["safe_mode"], bool)


class TestAutoDowngrade:
    def test_detailed_downgrade(self):
        """If VRAM is low, detailed should downgrade."""
        config = {
            "gpu_policy": {
                "min_free_vram_gb": 9999,  # Force safe mode
                "fallback_resolution": 512,
            },
            "machine": {"vram_gb": 6},
        }
        mode, warning = auto_downgrade_mode("detailed", config)
        assert mode == "balanced"
        assert "Downgraded" in warning


class TestGetDevice:
    def test_returns_string(self):
        device = get_device()
        assert device in ("cuda", "cpu")


class TestGetGpuName:
    def test_returns_string(self):
        name = get_gpu_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_nvml_name_supports_string_return(self, monkeypatch):
        import core.gpu as gpu

        monkeypatch.setattr(gpu, "_init_pynvml", lambda: True)
        monkeypatch.setattr(gpu.pynvml, "nvmlDeviceGetHandleByIndex", lambda _: object())
        monkeypatch.setattr(gpu.pynvml, "nvmlDeviceGetName", lambda _: "NVIDIA test GPU")

        assert gpu.get_gpu_name() == "NVIDIA test GPU"
