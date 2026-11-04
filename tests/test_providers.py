"""Tests for VLM providers."""
import pytest
import numpy as np
import httpx
import socket
from unittest.mock import patch

from providers.base import VlmProvider, VlmResponse, TokenUsage
from providers.lmstudio import LmStudioProvider
from providers.openai_compat import OpenAICompatProvider
from providers.llamacpp import LlamaCppProvider
from providers.cloud import CloudProvider
from providers.gemini import GeminiProvider
from providers.registry import ProviderRegistry, _extract_models


# ------------------------------------------------------------------
# Service availability cache (check once, skip fast if not running)
# ------------------------------------------------------------------

def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Quick TCP check — returns True if port is reachable."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


# Module-level cache: only checks each service once across all tests
_SERVICE_CACHE: dict[str, bool] = {}


def _is_available(name: str, host: str, port: int) -> bool:
    if name not in _SERVICE_CACHE:
        _SERVICE_CACHE[name] = _port_open(host, port)
    return _SERVICE_CACHE[name]


lmstudio_available = pytest.mark.skipif(
    not _is_available("lmstudio", "localhost", 1234),
    reason="LM Studio not running on :1234",
)
llamacpp_available = pytest.mark.skipif(
    not _is_available("llamacpp", "localhost", 8080),
    reason="llama.cpp not running on :8080",
)


# ------------------------------------------------------------------
# Abstract base class
# ------------------------------------------------------------------

class TestVlmProviderBase:
    def test_cannot_instantiate_abstract(self):
        """VlmProvider is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VlmProvider({"model": "test"})  # type: ignore[abstract]


# ------------------------------------------------------------------
# TokenUsage / VlmResponse dataclasses
# ------------------------------------------------------------------

class TestTokenUsage:
    def test_defaults(self):
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.reasoning_tokens == 0
        assert t.total_tokens == 0

    def test_custom_values(self):
        t = TokenUsage(prompt_tokens=10, completion_tokens=20, reasoning_tokens=5, total_tokens=30)
        assert t.prompt_tokens == 10
        assert t.completion_tokens == 20
        assert t.reasoning_tokens == 5
        assert t.total_tokens == 30


class TestVlmResponse:
    def test_defaults(self):
        r = VlmResponse(content="hello")
        assert r.content == "hello"
        assert isinstance(r.tokens, TokenUsage)

    def test_with_tokens(self):
        tokens = TokenUsage(prompt_tokens=5)
        r = VlmResponse(content="world", tokens=tokens)
        assert r.tokens.prompt_tokens == 5


# ------------------------------------------------------------------
# OpenAICompatProvider (shared base)
# ------------------------------------------------------------------

class TestOpenAICompatProvider:
    """Tests for the shared OpenAI-compatible base class."""

    def test_init(self):
        config = {
            "model": "test-model",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0.5,
            "max_tokens": 256,
            "timeout": 30,
        }
        provider = OpenAICompatProvider(config)
        try:
            assert provider.model_name == "test-model"
            assert provider.base_url == "http://localhost:1234/v1"
            assert provider.temperature == 0.5
            assert provider.max_tokens == 256
            assert provider.timeout == 30
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = OpenAICompatProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:1234/v1"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 1024
            assert provider.timeout == 60
        finally:
            provider.close()

    def test_api_key_sets_auth_header(self):
        config = {"api_key": "sk-test123"}
        provider = OpenAICompatProvider(config)
        try:
            assert provider._extra_headers["Authorization"] == "Bearer sk-test123"
        finally:
            provider.close()

    def test_no_api_key_no_auth_header(self):
        config = {"api_key": "not-needed"}
        provider = OpenAICompatProvider(config)
        try:
            assert "Authorization" not in provider._extra_headers
        finally:
            provider.close()

    def test_parse_usage(self):
        data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 10},
            }
        }
        usage = OpenAICompatProvider._parse_usage(data)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.reasoning_tokens == 10
        assert usage.total_tokens == 150

    def test_parse_usage_missing_fields(self):
        usage = OpenAICompatProvider._parse_usage({})
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_describe_frames_empty(self):
        provider = OpenAICompatProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_describe_frames_truncated(self):
        """More than 10 frames gets truncated."""
        provider = OpenAICompatProvider({})
        try:
            frames = [
                (float(i), np.ones((10, 10, 3), dtype=np.uint8) * 128)
                for i in range(15)
            ]
            mock_resp = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
            with patch.object(provider, "_chat_completion", return_value=mock_resp):
                result = provider.describe_frames(frames, "test")
            lines = result.split("\n")
            assert len(lines) == 11  # 10 frames + 1 truncation line
            assert any("truncated" in line for line in lines)
        finally:
            provider.close()

    def test_describe_frames_timestamps(self):
        """Frames are formatted as [MM:SS] in output."""
        provider = OpenAICompatProvider({})
        try:
            frames = [(65.0, np.ones((10, 10, 3), dtype=np.uint8) * 128)]
            mock_resp = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
            with patch.object(provider, "_chat_completion", return_value=mock_resp):
                result = provider.describe_frames(frames, "test")
            assert "[01:05]" in result
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "test",
            "timeout": 2,
        }
        provider = OpenAICompatProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content or "timed out" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = OpenAICompatProvider(
            {"base_url": "http://localhost:19999/v1", "timeout": 2}
        )
        try:
            assert provider.check_health() is False
        finally:
            provider.close()

    def test_list_models_bad_url(self):
        provider = OpenAICompatProvider(
            {"base_url": "http://localhost:19999/v1", "timeout": 2}
        )
        try:
            models = provider.list_models()
            assert models == []
        finally:
            provider.close()


# ------------------------------------------------------------------
# LM Studio provider
# ------------------------------------------------------------------

@lmstudio_available
class TestLmStudioProvider:
    """Unit / integration tests for LmStudioProvider."""

    def test_init(self):
        config = {
            "model": "google/gemma-4-e2b",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0.1,
            "max_tokens": 512,
        }
        provider = LmStudioProvider(config)
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:1234/v1"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 512
            assert provider.PROVIDER_NAME == "lmstudio"
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = LmStudioProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:1234/v1"
        finally:
            provider.close()

    def test_check_health_lmstudio_running(self):
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            health = provider.check_health()
            assert isinstance(health, bool)
        finally:
            provider.close()

    def test_describe_image(self):
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            if not provider.check_health():
                pytest.skip("LM Studio not running")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = provider.describe_image(img)
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_describe_image_with_question(self):
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            if not provider.check_health():
                pytest.skip("LM Studio not running")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = provider.describe_image(img, "What color is this?")
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_describe_frames(self):
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            if not provider.check_health():
                pytest.skip("LM Studio not running")
            frames = [(0.0, np.ones((100, 100, 3), dtype=np.uint8) * 128)]
            result = provider.describe_frames(frames, "What color is this?")
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            provider.close()

    def test_describe_frames_empty(self):
        provider = LmStudioProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_describe_frames_truncated(self):
        provider = LmStudioProvider({})
        try:
            frames = [
                (float(i), np.ones((10, 10, 3), dtype=np.uint8) * 128)
                for i in range(15)
            ]
            mock_resp = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
            with patch.object(provider, "_chat_completion", return_value=mock_resp):
                result = provider.describe_frames(frames, "test")
            lines = result.split("\n")
            assert len(lines) == 11
            assert any("truncated" in line for line in lines)
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "google/gemma-4-e2b",
            "timeout": 2,
        }
        provider = LmStudioProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = LmStudioProvider(
            {"base_url": "http://localhost:19999/v1", "timeout": 2}
        )
        try:
            assert provider.check_health() is False
        finally:
            provider.close()



# ------------------------------------------------------------------
# llama.cpp provider
# ------------------------------------------------------------------

@llamacpp_available
class TestLlamaCppProvider:
    """Tests for the llama.cpp provider (thin wrapper over OpenAICompatProvider)."""

    def test_init(self):
        config = {
            "model": "llava-13b",
            "base_url": "http://localhost:8080/v1",
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        provider = LlamaCppProvider(config)
        try:
            assert provider.model_name == "llava-13b"
            assert provider.base_url == "http://localhost:8080/v1"
            assert provider.temperature == 0.2
            assert provider.max_tokens == 1024
            assert provider.PROVIDER_NAME == "llamacpp"
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = LlamaCppProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:8080/v1"
        finally:
            provider.close()

    def test_describe_frames_empty(self):
        provider = LlamaCppProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "test",
            "timeout": 2,
        }
        provider = LlamaCppProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = LlamaCppProvider(
            {"base_url": "http://localhost:19999/v1", "timeout": 2}
        )
        try:
            assert provider.check_health() is False
        finally:
            provider.close()

    def test_health_check_llamacpp_running(self):
        """Integration test - llama.cpp should be reachable."""
        provider = LlamaCppProvider({})
        try:
            health = provider.check_health()
            assert isinstance(health, bool)
        finally:
            provider.close()

    def test_list_models_bad_url(self):
        provider = LlamaCppProvider(
            {"base_url": "http://localhost:19999/v1", "timeout": 2}
        )
        try:
            models = provider.list_models()
            assert models == []
        finally:
            provider.close()

    def test_describe_image(self):
        """Integration test - describe an image via llama.cpp."""
        provider = LlamaCppProvider({})
        try:
            if not provider.check_health():
                pytest.skip("llama.cpp not running")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = provider.describe_image(img)
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_query_text(self):
        """Integration test - text-only query to llama.cpp."""
        provider = LlamaCppProvider({})
        try:
            if not provider.check_health():
                pytest.skip("llama.cpp not running")
            result = provider.query_text("Say hello.")
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()


# ------------------------------------------------------------------
# Cloud provider (OpenAI, Groq, Together, vLLM, etc.)
# ------------------------------------------------------------------

class TestCloudProvider:
    """Tests for the generic OpenAI-compatible provider (OpenAI, Groq, vLLM, etc.)."""

    def test_init(self):
        config = {
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "temperature": 0.5,
            "max_tokens": 2048,
        }
        provider = CloudProvider(config)
        try:
            assert provider.model_name == "gpt-4o"
            assert provider.base_url == "https://api.openai.com/v1"
            assert provider.temperature == 0.5
            assert provider.max_tokens == 2048
            assert provider.PROVIDER_NAME == "openai"
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = CloudProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            # Falls back to "not-needed" when no env var set
            assert provider._extra_headers.get("Authorization") is None
        finally:
            provider.close()

    def test_env_var_fallback(self, monkeypatch):
        """API key falls back to OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        provider = CloudProvider({"model": "test"})
        try:
            assert provider._extra_headers["Authorization"] == "Bearer sk-env-test"
        finally:
            provider.close()

    def test_config_key_takes_precedence(self, monkeypatch):
        """Config api_key takes precedence over env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        provider = CloudProvider({"api_key": "sk-config-test"})
        try:
            assert provider._extra_headers["Authorization"] == "Bearer sk-config-test"
        finally:
            provider.close()

    def test_describe_frames_empty(self):
        provider = CloudProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "test",
            "timeout": 2,
        }
        provider = CloudProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content or "timed out" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = CloudProvider(
            {"base_url": "http://localhost:19999/v1", "timeout": 2}
        )
        try:
            assert provider.check_health() is False
        finally:
            provider.close()

    def test_list_models_bad_url(self):
        provider = CloudProvider(
            {"base_url": "http://localhost:19999/v1", "timeout": 2}
        )
        try:
            models = provider.list_models()
            assert models == []
        finally:
            provider.close()


# ------------------------------------------------------------------
# ProviderRegistry
# ------------------------------------------------------------------

class TestProviderRegistry:
    """Tests for the provider registry and factory."""

    def test_create_explicit_lmstudio(self):
        config = {"provider": "lmstudio", "model": "test-model"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, LmStudioProvider)
            assert provider.model_name == "test-model"
        finally:
            provider.close()

    def test_create_explicit_gemini(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        config = {"provider": "gemini", "model": "gemini-3.1-flash-lite"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, GeminiProvider)
            assert provider.model_name == "gemini-3.1-flash-lite"
        finally:
            provider.close()

    def test_create_explicit_llamacpp(self):
        config = {"provider": "llamacpp", "model": "llava"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, LlamaCppProvider)
            assert provider.model_name == "llava"
        finally:
            provider.close()

    def test_create_explicit_openai(self):
        config = {"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, CloudProvider)
            assert provider.model_name == "gpt-4o"
        finally:
            provider.close()

    def test_create_explicit_groq(self):
        config = {"provider": "groq", "model": "llava", "api_key": "sk-test"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, CloudProvider)
        finally:
            provider.close()

    def test_create_explicit_together(self):
        config = {"provider": "together", "model": "llava", "api_key": "sk-test"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, CloudProvider)
        finally:
            provider.close()

    def test_create_explicit_vllm(self):
        config = {"provider": "vllm", "model": "llava", "api_key": "not-needed"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, CloudProvider)
        finally:
            provider.close()

    def test_create_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderRegistry.create({"provider": "nonexistent"})

    def test_create_auto_with_mock(self, monkeypatch):
        """Auto-detect falls back to first detected provider."""
        # Mock detect to return a known provider
        monkeypatch.setattr(
            ProviderRegistry, "detect",
            lambda config=None: [{"name": "lmstudio", "config": {"base_url": "http://test", "model": "m"}}],
        )
        provider = ProviderRegistry.create({"provider": "auto"})
        try:
            assert isinstance(provider, LmStudioProvider)
        finally:
            provider.close()

    def test_create_auto_no_providers(self, monkeypatch):
        """Auto-detect with no running providers raises ValueError."""
        monkeypatch.setattr(ProviderRegistry, "detect", lambda config=None: [])
        with pytest.raises(ValueError, match="No VLM provider found"):
            ProviderRegistry.create({"provider": "auto"})

    def test_list_providers(self):
        providers = ProviderRegistry.list_providers()
        assert "lmstudio" in providers
        assert "gemini" in providers
        assert "ollama" not in providers
        assert "llamacpp" in providers
        assert "openai" in providers
        assert "groq" in providers

    def test_detect_returns_list(self, monkeypatch):
        """detect() returns a list even with mocked endpoints."""
        import httpx
        def mock_get(url, **kwargs):
            raise httpx.ConnectError("mock")
        monkeypatch.setattr(httpx, "get", mock_get)
        result = ProviderRegistry.detect()
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_detect_result_structure(self, monkeypatch):
        import httpx
        def mock_get(url, **kwargs):
            raise httpx.ConnectError("mock")
        monkeypatch.setattr(httpx, "get", mock_get)
        result = ProviderRegistry.detect()
        for item in result:
            assert "name" in item
            assert "status" in item
            assert item["status"] in ("available", "unavailable", "timeout", "error")

    def test_lmstudio_defaults_set(self, monkeypatch):
        # No running server: the default fallback port is 1234.
        monkeypatch.setattr(ProviderRegistry, "detect", lambda config=None: [])
        config = {"provider": "lmstudio"}
        provider = ProviderRegistry.create(config)
        try:
            assert provider.base_url == "http://localhost:1234/v1"
        finally:
            provider.close()

    def test_gemini_defaults_set(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        provider = ProviderRegistry.create({"provider": "gemini"})
        try:
            assert isinstance(provider, GeminiProvider)
            assert provider.base_url == "https://generativelanguage.googleapis.com/v1beta"
            assert provider.model_name == "gemini-3.1-flash-lite"
        finally:
            provider.close()

    def test_llamacpp_defaults_set(self):
        config = {"provider": "llamacpp"}
        provider = ProviderRegistry.create(config)
        try:
            assert provider.base_url == "http://localhost:8080/v1"
        finally:
            provider.close()


# ------------------------------------------------------------------
# _extract_models helper
# ------------------------------------------------------------------

class TestExtractModels:
    def test_ollama_shape_removed(self):
        data = {"models": [{"name": "llava:7b"}, {"name": "mistral"}]}
        assert _extract_models("ollama", data) == []

    def test_lmstudio_format(self):
        data = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        models = _extract_models("lmstudio", data)
        assert models == ["model-a", "model-b"]

    def test_llamacpp_format(self):
        data = {"data": [{"id": "ggml-model"}]}
        models = _extract_models("llamacpp", data)
        assert models == ["ggml-model"]

    def test_unknown_provider(self):
        models = _extract_models("unknown", {})
        assert models == []

    def test_empty_data(self):
        assert _extract_models("ollama", {}) == []
        assert _extract_models("lmstudio", {}) == []
