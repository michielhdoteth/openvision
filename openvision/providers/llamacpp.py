"""llama.cpp server VLM provider via OpenAI-compatible API.

llama.cpp serves an OpenAI-compatible /v1/chat/completions endpoint.
This is a thin wrapper around OpenAICompatProvider with llama.cpp defaults.
"""
from openvision.providers.openai_compat import OpenAICompatProvider


class LlamaCppProvider(OpenAICompatProvider):
    """llama.cpp server provider using OpenAI-compatible API.

    Connects to a local llama.cpp server at ``base_url`` (default
    ``http://localhost:8080/v1``) and uses its /v1/chat/completions endpoint.

    llama.cpp must be started with a vision-capable model (e.g. llava).
    """

    PROVIDER_NAME = "llamacpp"

    def __init__(self, config: dict):
        config.setdefault("base_url", "http://localhost:8080/v1")
        config.setdefault("api_key", "not-needed")
        super().__init__(config)
