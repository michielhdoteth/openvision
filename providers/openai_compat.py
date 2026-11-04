"""Shared base class for OpenAI-compatible VLM providers.

This module provides OpenAICompatProvider, a reusable base for any provider
that speaks the OpenAI chat completions protocol with vision (base64 images).
Concrete providers (LM Studio, llama.cpp, OpenAI, Groq, vLLM, Together)
extend this and just configure base_url, model name, and auth headers.
"""
from typing import Optional
import base64
import re

import httpx
import numpy as np

from providers.base import VlmProvider, VlmResponse, TokenUsage
from core.image import encode_base64


# Error text a text-only model produces when given image input. These arrive
# as HTTP 4xx/5xx from llama.cpp-style servers started without a multimodal
# projector, or from chat-only hosted models.
_IMAGE_REJECTED = re.compile(
    r"image input is not supported|failed to (load|format) image|"
    r"images? (are|is) not supported|does not support (image|vision|multimodal)",
    re.IGNORECASE,
)


class OpenAICompatProvider(VlmProvider):
    """OpenAI-compatible chat completions provider with vision support.

    Subclasses only need to set ``self.base_url``, ``self.model_name``,
    and optionally ``self._extra_headers`` before calling ``super().__init__()``.

    Config keys consumed:
        base_url: API base URL (e.g. http://localhost:1234/v1)
        model: Model identifier
        api_key: Optional API key (default: "not-needed" for local providers)
        temperature: Sampling temperature (default: 0.1)
        max_tokens: Max response tokens (default: 1024)
        timeout: HTTP timeout in seconds (default: 60)
    """

    PROVIDER_NAME: str = "openai_compat"  # Override in subclasses

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:1234/v1")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1024)
        self.timeout = config.get("timeout", 60)
        api_key = config.get("api_key", "not-needed")
        self._extra_headers: dict = {}
        if api_key and api_key != "not-needed":
            self._extra_headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            timeout=self.timeout,
            headers=self._extra_headers,
        )
        self._image_support_cache: dict = {}


    @staticmethod
    def _parse_usage(data: dict) -> TokenUsage:
        """Extract token usage from OpenAI-compatible response."""
        usage = data.get("usage", {})
        details = usage.get("completion_tokens_details", {})
        return TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def _chat_completion(self, messages: list[dict], max_tokens: Optional[int] = None) -> dict:
        """Send a chat completion request. Returns raw response dict."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def _error_response(self, exc: Exception, context: str = "provider") -> VlmResponse:
        """Create a VlmResponse from an exception."""
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return VlmResponse(
                content=(
                    f"Could not connect to {context} at {self.base_url}: {exc}. "
                    "Is the server running?"
                ),
                tokens=TokenUsage(),
            )
        elif isinstance(exc, httpx.TimeoutException):
            return VlmResponse(
                content=(
                    f"Request to {context} timed out after {self.timeout}s. "
                    "The model may be overloaded."
                ),
                tokens=TokenUsage(),
            )
        elif isinstance(exc, httpx.HTTPStatusError):
            return VlmResponse(
                content=(
                    f"{context} returned HTTP {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                ),
                tokens=TokenUsage(),
            )
        else:
            return VlmResponse(
                content=f"Unexpected error contacting {context}: {exc}",
                tokens=TokenUsage(),
            )


    def describe_image(
        self, image: np.ndarray, question: Optional[str] = None
    ) -> VlmResponse:
        """Send an image to the provider and get a description."""
        b64_image = encode_base64(image)

        if question:
            text = (
                f"Answer this question about the image: {question}\n"
                "Be concise and specific. Read all numbers, labels, and text carefully."
            )
        else:
            text = (
                "Describe this image in detail. "
                "If it contains a chart, graph, or data visualization, "
                "read ALL numbers, labels, axis values, and data points precisely. "
                "List the main objects visible."
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": b64_image},
                    },
                ],
            }
        ]

        try:
            data = self._chat_completion(messages)
            message = data["choices"][0]["message"]
            tokens = self._parse_usage(data)

            # Reasoning models spend the budget thinking and can finish with an
            # empty content. Read content first, always; only fall back when it
            # is genuinely empty AND reasoning explains why.
            content = (message.get("content") or "").strip()
            if not content and \
                    (message.get("reasoning_content") or "").strip() and \
                    data["choices"][0].get("finish_reason") == "length":
                retry = self._chat_completion(messages, max_tokens=self.max_tokens + 1024)
                choice = retry["choices"][0]
                message = choice["message"]
                tokens = self._parse_usage(retry)
                content = (message.get("content") or "").strip()
                if not content:
                    spent = retry.get("usage", {}).get(
                        "completion_tokens_details", {}).get("reasoning_tokens", "the")
                    return VlmResponse(
                        content=(
                            f"Model {self.model_name} spent {spent} token budget on "
                            f"reasoning and produced no answer. Raise --max-tokens "
                            f"or load a non-thinking model."
                        ),
                        tokens=tokens,
                    )
            return VlmResponse(content=content, tokens=tokens)
        except Exception as exc:
            if self._image_rejected(exc):
                return VlmResponse(
                    content=(
                        f"Model {self.model_name} at {self.base_url} does not accept "
                        f"image input. Load a vision-capable model with its projector "
                        f"(mmproj), or run with --raw-frames so a vision-capable agent "
                        f"can read the frames."
                    ),
                    tokens=TokenUsage(),
                )
            return self._error_response(exc, self.PROVIDER_NAME)

    def describe_frames(
        self,
        frames: list[tuple[float, np.ndarray]],
        question: Optional[str] = None,
    ) -> str:
        """Analyze multiple frames from a video timeline."""
        if not frames:
            return "No frames to analyze."

        results: list[str] = []
        for timestamp, image in frames[:10]:
            resp = self.describe_image(image, question)
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            results.append(f"[{mins:02d}:{secs:02d}] {resp.content}")

        if len(frames) > 10:
            results.append(
                f"... and {len(frames) - 10} more frames (truncated)"
            )

        return "\n".join(results)

    def query_text(self, prompt: str) -> VlmResponse:
        """Send a text-only query (no image)."""
        messages = [{"role": "user", "content": prompt}]
        try:
            data = self._chat_completion(messages)
            content = data["choices"][0]["message"]["content"].strip()
            tokens = self._parse_usage(data)
            return VlmResponse(content=content, tokens=tokens)
        except Exception as exc:
            return self._error_response(exc, self.PROVIDER_NAME)

    def check_health(self) -> bool:
        """Check if the provider API is responsive."""
        try:
            response = self._client.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def supports_images(self) -> bool:
        """Detect image support from the model listing, probing if undeclared.

        llama.cpp-style listings (LM Studio) do not report input modalities,
        so a 1x1 pixel is sent and the answer is cached per provider+model.
        A declared "text"-only capability short-circuits without a round trip.
        """
        cache_key = (self.base_url, self.model_name)
        if cache_key in self._image_support_cache:
            return self._image_support_cache[cache_key]

        supported = self._detect_image_support()
        self._image_support_cache[cache_key] = supported
        return supported

    def _detect_image_support(self) -> bool:
        try:
            response = self._client.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                for model in response.json().get("data", []):
                    if model.get("id") != self.model_name:
                        continue
                    modalities = (model.get("architecture") or {}).get("input_modalities")
                    if modalities:
                        return "image" in modalities or "vision" in modalities
        except Exception:
            pass

        # Undeclared modalities: ask the model with one invisible pixel. A
        # rejection means the model or its server has no image path.
        try:
            probe = np.zeros((1, 1, 3), dtype=np.uint8)
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {"type": "image_url", "image_url": {"url": encode_base64(probe, ".png")}},
                        ],
                    }],
                    "max_tokens": 1,
                    "temperature": 0,
                },
                timeout=30,
            )
            # llama.cpp-style servers may answer 200 with an error body; treat
            # any non-2xx or an image-rejection message as "no image support".
            if response.status_code >= 400:
                return False
            if _IMAGE_REJECTED.search(response.text or ""):
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _image_rejected(exc: Exception) -> bool:
        """True when an error looks like the model refusing image input."""
        if isinstance(exc, httpx.HTTPStatusError):
            return bool(_IMAGE_REJECTED.search(exc.response.text or ""))
        return bool(_IMAGE_REJECTED.search(str(exc)))

    def list_models(self) -> list[str]:
        """List available models from the provider."""
        try:
            response = self._client.get(f"{self.base_url}/models", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return []


    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
