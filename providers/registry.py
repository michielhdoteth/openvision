"""Provider registry and factory for VLM providers.

Auto-discovers available providers and creates instances based on config.
Supports explicit provider selection and automatic detection.
"""
from typing import Optional
from providers.base import VlmProvider


# Provider class registry (lazy imports to avoid circular deps)
_PROVIDERS: dict[str, type[VlmProvider]] = {}


def _get_providers() -> dict[str, type[VlmProvider]]:
    """Lazy-load provider classes to avoid import overhead."""
    if not _PROVIDERS:
        from providers.lmstudio import LmStudioProvider
        from providers.ollama import OllamaProvider
        from providers.llamacpp import LlamaCppProvider
        from providers.cloud import CloudProvider

        _PROVIDERS.update({
            "lmstudio": LmStudioProvider,
            "lmstudio-local": LmStudioProvider,
            "ollama": OllamaProvider,
            "llamacpp": LlamaCppProvider,
            "openai": CloudProvider,
            "groq": CloudProvider,
            "together": CloudProvider,
            "vllm": CloudProvider,
            "localai": CloudProvider,
        })
    return _PROVIDERS


# Default endpoints for auto-detection
_DEFAULT_ENDPOINTS = [
    ("lmstudio", "http://localhost:1234/v1/models"),
    ("ollama", "http://localhost:11434/api/tags"),
    ("llamacpp", "http://localhost:8080/v1/models"),
]


class ProviderRegistry:
    """Auto-discover and instantiate VLM providers.

    Usage:
        # Explicit provider
        provider = ProviderRegistry.create({"provider": "ollama", "model": "llava:13b"})

        # Auto-detect first available
        provider = ProviderRegistry.create({"provider": "auto"})

        # Probe all endpoints
        available = ProviderRegistry.detect()
    """

    @classmethod
    def create(cls, config: dict) -> VlmProvider:
        """Create a VLM provider from config.

        Config keys:
            provider: Provider name ("auto", "lmstudio", "ollama", etc.)
            model: Model name (provider-specific)
            base_url: Override base URL
            api_key: API key (for remote providers)

        Returns:
            VlmProvider instance.

        Raises:
            ValueError: If provider is unknown or unavailable.
        """
        import httpx

        provider_name = config.get("provider", "auto").lower()

        if provider_name == "auto":
            detected = cls.detect()
            if not detected:
                raise ValueError(
                    "No VLM provider found. Start LM Studio, Ollama, or llama.cpp, "
                    "or configure a remote provider in ~/.openvision/config.yaml"
                )
            provider_name = detected[0]["name"]
            # Merge detected config
            config = {**detected[0].get("config", {}), **config}
            config["provider"] = provider_name
        elif config.get("model", "auto") == "auto":
            # Resolve the model sentinel even when the caller chose a provider.
            matching = next((item for item in cls.detect() if item.get("name") == provider_name), None)
            if matching and matching.get("models"):
                config = {**matching.get("config", {}), **config, "model": matching["models"][0]}

        providers = _get_providers()

        if provider_name not in providers:
            available = ", ".join(sorted(providers.keys()))
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {available}"
            )

        provider_cls = providers[provider_name]

        # Set provider-specific defaults
        if provider_name == "lmstudio" and "base_url" not in config:
            config["base_url"] = "http://localhost:1234/v1"
        elif provider_name == "ollama" and "base_url" not in config:
            config["base_url"] = "http://localhost:11434"
        elif provider_name == "llamacpp" and "base_url" not in config:
            config["base_url"] = "http://localhost:8080/v1"

        return provider_cls(config)

    @classmethod
    def detect(cls) -> list[dict]:
        """Probe common endpoints and return available providers.

        Returns:
            List of dicts with "name", "url", "status", "models" keys.
            Sorted by response time (fastest first).
        """
        import httpx
        import time

        results = []

        for name, url in _DEFAULT_ENDPOINTS:
            start = time.monotonic()
            try:
                response = httpx.get(url, timeout=3)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                if response.status_code == 200:
                    data = response.json()
                    models = _extract_models(name, data)
                    results.append({
                        "name": name,
                        "url": url.rsplit("/models", 1)[0].rsplit("/api/tags", 1)[0],
                        "status": "available",
                        "models": models,
                        "latency_ms": elapsed_ms,
                        "config": {
                            "base_url": url.rsplit("/models", 1)[0].rsplit("/api/tags", 1)[0],
                            "model": models[0] if models else "auto",
                        },
                    })
                else:
                    results.append({
                        "name": name,
                        "url": url,
                        "status": "error",
                        "error": f"HTTP {response.status_code}",
                        "latency_ms": elapsed_ms,
                    })
            except httpx.ConnectError:
                results.append({
                    "name": name,
                    "url": url,
                    "status": "unavailable",
                    "error": "Connection refused",
                })
            except httpx.TimeoutException:
                results.append({
                    "name": name,
                    "url": url,
                    "status": "timeout",
                    "error": "Connection timed out",
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "url": url,
                    "status": "error",
                    "error": str(e),
                })

        # Sort: available first, then by latency
        results.sort(key=lambda r: (
            0 if r["status"] == "available" else 1,
            r.get("latency_ms", 9999),
        ))

        return results

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return sorted(_get_providers().keys())


def _extract_models(provider_name: str, data: dict) -> list[str]:
    """Extract model names from provider-specific response format."""
    if provider_name == "ollama":
        return [m.get("name", "") for m in data.get("models", [])]
    elif provider_name in ("lmstudio", "llamacpp"):
        return [m.get("id", "") for m in data.get("data", [])]
    return []
