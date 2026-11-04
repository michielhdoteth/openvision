"""Provider registry and factory for VLM providers.

Auto-discovers available providers and creates instances based on config.
Supports explicit provider selection and automatic detection.
"""
import os
from typing import Optional
from providers.base import VlmProvider


# Provider class registry (lazy imports to avoid circular deps)
_PROVIDERS: dict[str, type] = {}

# Aliases that pin a decision-model preset and model id. Config values still
# win, except an "auto" model sentinel, which lets the alias pick.
_DECISION_ALIASES: dict[str, dict] = {
    "decisions": {"preset": "openrouter", "model": "openai/gpt-6-luna-decisions"},
    "openrouter-decisions": {"preset": "openrouter", "model": "openai/gpt-6-luna-decisions"},
    "luna-decisions": {"preset": "openrouter", "model": "openai/gpt-6-luna-decisions"},
    "clef": {"preset": "openrouter", "model": "cloudflare/clef"},
    "clef-flash": {"preset": "openrouter", "model": "cloudflare/clef-flash"},
    "clef-omni": {"preset": "openrouter", "model": "cloudflare/clef-omni"},
    "jev": {"preset": "openrouter", "model": "typesafe/jev-1.14"},
    "zen": {"preset": "zen", "model": "jev-1.13-free"},
    "jev-zen": {"preset": "zen", "model": "jev-1.13-free"},
    "openai-decisions": {"preset": "openai", "model": "gpt-6-luna"},
}


def _get_providers() -> dict[str, type]:
    """Lazy-load provider classes to avoid import overhead."""
    if not _PROVIDERS:
        from providers.lmstudio import LmStudioProvider
        from providers.llamacpp import LlamaCppProvider
        from providers.cloud import CloudProvider
        from providers.gemini import GeminiProvider
        from providers.decisions import DecisionProvider

        _PROVIDERS.update({
            "lmstudio": LmStudioProvider,
            "lmstudio-local": LmStudioProvider,
            "llamacpp": LlamaCppProvider,
            "gemini": GeminiProvider,
            "google": GeminiProvider,
            "openai": CloudProvider,
            "groq": CloudProvider,
            "together": CloudProvider,
            "vllm": CloudProvider,
            "localai": CloudProvider,
        })
        _PROVIDERS.update({name: DecisionProvider for name in _DECISION_ALIASES})
    return _PROVIDERS


# Default endpoints for auto-detection (local servers only; Gemini is
# keyed and remote, so it is selected explicitly rather than probed). A
# configured base_url is probed ahead of these, so a non-default port only
# needs to be set once in config or via OPENVISION_VLM_BASE_URL.
_DEFAULT_ENDPOINTS = [
    ("lmstudio", "http://localhost:1234/v1/models"),
    ("lmstudio", "http://localhost:1235/v1/models"),
    ("llamacpp", "http://localhost:8080/v1/models"),
]


class ProviderRegistry:
    """Auto-discover and instantiate VLM providers.

    Usage:
        # Explicit provider
        provider = ProviderRegistry.create({"provider": "llamacpp", "model": "qwen2.5-vl"})

        # Auto-detect first available
        provider = ProviderRegistry.create({"provider": "auto"})

        # Probe all endpoints
        available = ProviderRegistry.detect()
    """

    @classmethod
    def create(cls, config: dict) -> VlmProvider:
        """Create a VLM provider from config.

        Config keys:
            provider: Provider name ("auto", "lmstudio", "llamacpp", "gemini", "openai", ...)
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
        config = cls._resolve_connection(config, provider_name)

        if provider_name == "auto":
            detected = cls.detect(config)
            if not detected:
                raise ValueError(
                    "No VLM provider found. Start LM Studio or llama.cpp, "
                    "or configure a remote provider (openai, groq, together, "
                    "gemini) in ~/.openvision/config.yaml"
                )
            provider_name = detected[0]["name"]
            # Use the detected endpoint's config entirely. The user's "auto"
            # config carries no base_url or model to preserve.
            config = {**detected[0].get("config", {})}
            config["provider"] = provider_name
        elif config.get("model", "auto") == "auto":
            # Resolve the model sentinel. If the user configured a base_url,
            # probe that URL for the model list. Otherwise probe the default
            # endpoints. Never let a detected endpoint override an explicitly
            # configured base_url.
            configured_url = config.get("base_url")
            if configured_url:
                matching = cls._probe_one(provider_name, configured_url)
            else:
                matching = next((item for item in cls.detect(config)
                                 if item.get("name") == provider_name), None)
            if matching and matching.get("models"):
                config["model"] = matching["models"][0]
                if not configured_url and matching.get("config", {}).get("base_url"):
                    config["base_url"] = matching["config"]["base_url"]

        providers = _get_providers()

        if provider_name not in providers:
            available = ", ".join(sorted(providers.keys()))
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {available}"
            )

        provider_cls = providers[provider_name]

        # Decision aliases pin a preset and model; explicit config still wins
        # unless the config only carries the "auto" sentinel.
        if provider_name in _DECISION_ALIASES:
            merged = dict(_DECISION_ALIASES[provider_name])
            merged.update({
                key: value
                for key, value in config.items()
                if not (key == "model" and value == "auto")
            })
            merged["provider"] = provider_name
            config = merged

        # Fall back to the shipped per-provider default. A base_url supplied by
        # config, flag, or env was already merged in by _resolve_connection.
        local_defaults = {
            "lmstudio": "http://localhost:1234/v1",
            "llamacpp": "http://localhost:8080/v1",
        }
        if provider_name in local_defaults and not config.get("base_url"):
            config["base_url"] = local_defaults[provider_name]

        return provider_cls(config)

    # Providers whose endpoint lives on a local port, with the URL shape the
    # probe list expects.
    _LOCAL_PROVIDERS = {
        "lmstudio": "http://localhost:{port}/v1/models",
        "llamacpp": "http://localhost:{port}/v1/models",
    }

    @classmethod
    def _resolve_connection(cls, config: dict, provider_name: str) -> dict:
        """Merge connection overrides into provider config.

        Precedence: top-level keys (flag or direct config) > env var > the
        per-provider nested block documented in config/default.yaml. Without
        this, models.vlm.<provider>.base_url never reached the provider, so
        running LM Studio on any non-default port was unreachable.
        """
        resolved = dict(config)

        nested = resolved.get(provider_name) if provider_name else None
        if isinstance(nested, dict):
            for key in ("base_url", "model"):
                if nested.get(key) and not resolved.get(key):
                    resolved[key] = nested[key]
            resolved.pop(provider_name, None)

        env_url = os.environ.get("OPENVISION_VLM_BASE_URL")
        env_model = os.environ.get("OPENVISION_VLM_MODEL")
        if env_url and not resolved.get("base_url"):
            resolved["base_url"] = env_url
        if env_model and (not resolved.get("model") or resolved.get("model") == "auto"):
            resolved["model"] = env_model

        return resolved

    @classmethod
    def _candidate_endpoints(cls, config: Optional[dict] = None) -> list[tuple[str, str]]:
        """Build the endpoint probe list: configured first, then defaults."""
        endpoints: list[tuple[str, str]] = []
        seen: set[str] = set()
        config = config or {}
        base_url = config.get("base_url")
        if base_url:
            url = base_url.rstrip("/")
            if not url.endswith("/models"):
                url = f"{url}/models"
            name = config.get("provider", "lmstudio")
            if name not in cls._LOCAL_PROVIDERS:
                name = "lmstudio"
            endpoints.append((name, url))
            seen.add(url)
        for name, url in _DEFAULT_ENDPOINTS:
            if url not in seen:
                endpoints.append((name, url))
                seen.add(url)
        return endpoints

    @classmethod
    def _probe_one(cls, name: str, base_url: str) -> Optional[dict]:
        """Probe a single configured endpoint so the rest of the flow works."""
        import httpx

        url = base_url.rstrip("/")
        if not url.endswith("/models"):
            url = f"{url}/models"
        try:
            response = httpx.get(url, timeout=3)
            if response.status_code != 200:
                return None
            models = _extract_models(name, response.json())
            return {
                "name": name,
                "url": url.rsplit("/models", 1)[0],
                "status": "available",
                "models": models,
                "config": {
                    "base_url": url.rsplit("/models", 1)[0],
                    "model": models[0] if models else "auto",
                },
            }
        except Exception:
            return None

    @classmethod
    def detect(cls, config: Optional[dict] = None) -> list[dict]:
        """Probe common endpoints and return available providers.

        A configured base_url is probed first, so "auto" finds a local server
        on a non-default port without needing the default list to match.

        Returns:
            List of dicts with "name", "url", "status", "models" keys.
            Sorted by response time (fastest first).
        """
        import httpx
        import time
        from concurrent.futures import ThreadPoolExecutor

        endpoints = cls._candidate_endpoints(config)

        def probe(endpoint):
            name, url = endpoint
            start = time.monotonic()
            try:
                response = httpx.get(url, timeout=3)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                if response.status_code == 200:
                    data = response.json()
                    models = _extract_models(name, data)
                    return {
                        "name": name,
                        "url": url.rsplit("/models", 1)[0].rsplit("/api/tags", 1)[0],
                        "status": "available",
                        "models": models,
                        "latency_ms": elapsed_ms,
                        "config": {
                            "base_url": url.rsplit("/models", 1)[0].rsplit("/api/tags", 1)[0],
                            "model": models[0] if models else "auto",
                        },
                    }
                return {
                    "name": name,
                    "url": url,
                    "status": "error",
                    "error": f"HTTP {response.status_code}",
                    "latency_ms": elapsed_ms,
                }
            except httpx.ConnectError:
                return {
                    "name": name,
                    "url": url,
                    "status": "unavailable",
                    "error": "Connection refused",
                }
            except httpx.TimeoutException:
                return {
                    "name": name,
                    "url": url,
                    "status": "timeout",
                    "error": "Connection timed out",
                }
            except Exception as e:
                return {
                    "name": name,
                    "url": url,
                    "status": "error",
                    "error": str(e),
                }

        # Probing sequentially made every detect() caller wait out each timeout
        # one after another; the endpoints are independent, so ask in parallel.
        with ThreadPoolExecutor(max_workers=max(1, len(endpoints))) as pool:
            results = list(pool.map(probe, endpoints))

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
    if provider_name in ("lmstudio", "llamacpp"):
        return [m.get("id", "") for m in data.get("data", [])]
    return []
