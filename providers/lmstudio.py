"""LM Studio VLM provider via OpenAI-compatible HTTP API.

Thin wrapper around OpenAICompatProvider with LM Studio defaults.
"""
from providers.openai_compat import OpenAICompatProvider


class LmStudioProvider(OpenAICompatProvider):
    """LM Studio provider using the OpenAI-compatible chat completions endpoint.

    Connects to a local LM Studio instance at ``base_url`` (default
    ``http://localhost:1234/v1``) and uses its /v1/chat/completions endpoint
    with ``image_url`` content blocks that carry base64-encoded images.
    """

    PROVIDER_NAME = "lmstudio"

    def __init__(self, config: dict):
        config.setdefault("base_url", "http://localhost:1234/v1")
        config.setdefault("api_key", "not-needed")
        super().__init__(config)
