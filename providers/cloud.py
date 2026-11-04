"""Cloud VLM provider — remote API backends.

Works with OpenAI, Groq, vLLM, Together AI, LocalAI, and any other
provider that implements the OpenAI chat completions protocol with vision.
"""
import os
from typing import Optional

from providers.openai_compat import OpenAICompatProvider


# Per-backend env var fallbacks for the API key (checked in this order).
_ENV_KEYS: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "groq": ["GROQ_API_KEY", "OPENAI_API_KEY"],
    "together": ["TOGETHER_API_KEY", "OPENAI_API_KEY"],
    "vllm": ["VLLM_API_KEY"],
    "localai": ["LOCALAI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


class CloudProvider(OpenAICompatProvider):
    """Remote/cloud provider for any OpenAI-compatible API with vision support.

    Supports: OpenAI, Groq, vLLM, Together AI, LocalAI, Anyscale, etc.

    Config keys consumed:
        base_url: API base URL (required for non-OpenAI providers)
        model: Model identifier (required)
        api_key: API key (or the provider's env var, see _ENV_KEYS)
        temperature: Sampling temperature (default: 0.1)
        max_tokens: Max response tokens (default: 1024)
        timeout: HTTP timeout in seconds (default: 60)
    """

    PROVIDER_NAME = "openai"

    def __init__(self, config: dict):
        backend = (config.get("provider") or self.PROVIDER_NAME).lower()
        env_names = _ENV_KEYS.get(backend, _ENV_KEYS["openai"])
        api_key = config.get("api_key")
        if not api_key:
            for name in env_names:
                api_key = os.environ.get(name, "")
                if api_key:
                    break
        if not api_key:
            api_key = "not-needed"  # Allow unauthenticated local servers

        config["api_key"] = api_key
        super().__init__(config)


GenericOpenAIProvider = CloudProvider
