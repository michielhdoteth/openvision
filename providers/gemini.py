"""Google Gemini provider — native generateContent API.

Gemini is the only backend wired here that accepts **video** input: video is
sent as a ``fileData`` part (``fileUri`` + ``mimeType``), a public HTTP(S) URL,
or a ``gs://`` URI, alongside the text prompt. Images go as ``inline_data``
base64 parts; text-only queries skip media entirely.

API surface is the REST ``generateContent`` endpoint, so no Google SDK
dependency is required. Auth uses the ``x-goog-api-key`` header.

Config keys consumed:
    model: Model identifier (default "gemini-3.1-flash-lite" — cheapest
           verified video-input tier at ~ $0.25 / $1.50 per 1M in/out)
    base_url: API base (default "https://generativelanguage.googleapis.com/v1beta";
              env GEMINI_BASE_URL overrides). Point this at a Vertex-hosted
              gateway or proxy if needed.
    api_key: Google API key (config > GEMINI_API_KEY > GOOGLE_API_KEY env)
    temperature: Sampling temperature (default 0.1)
    max_tokens: Max response tokens (default 1024)
    timeout: HTTP timeout in seconds (default 120 — video calls are slow)
"""
from __future__ import annotations

import os
from typing import Optional

import httpx
import numpy as np

from core.image import encode_base64
from providers.base import VlmProvider, VlmResponse, TokenUsage


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GeminiProvider(VlmProvider):
    """Gemini provider over the native generateContent REST API."""

    PROVIDER_NAME = "gemini"

    def __init__(self, config: dict):
        super().__init__(config)
        # "auto" (the registry's unresolved sentinel) falls back to the
        # cheapest verified video-input tier.
        self.model_name = config.get("model") if config.get("model") not in (None, "", "auto") else DEFAULT_MODEL
        self.base_url = (
            config.get("base_url")
            or os.environ.get("GEMINI_BASE_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            config.get("api_key")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1024)
        self.timeout = config.get("timeout", 120)
        self._client = httpx.Client(
            timeout=self.timeout,
            headers={"x-goog-api-key": self.api_key} if self.api_key else {},
        )


    def _generate(self, parts: list[dict], context: str = "Gemini") -> VlmResponse:
        """POST contents to generateContent and normalize the response."""
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        try:
            response = self._client.post(
                f"{self.base_url}/models/{self.model_name}:generateContent",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 — normalize into VlmResponse
            return self._error_response(exc, context)

        text_parts = []
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if part.get("text"):
                    text_parts.append(part["text"])
        usage = data.get("usageMetadata", {}) or {}
        return VlmResponse(
            content="\n".join(text_parts).strip(),
            tokens=TokenUsage(
                prompt_tokens=usage.get("promptTokenCount", 0),
                completion_tokens=usage.get("candidatesTokenCount", 0),
                reasoning_tokens=usage.get("thoughtsTokenCount", 0),
                total_tokens=usage.get("totalTokenCount", 0),
            ),
        )

    def _error_response(self, exc: Exception, context: str = "Gemini") -> VlmResponse:
        """Create a VlmResponse from an exception, mirroring OpenAI-compat."""
        if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
            return VlmResponse(
                content=(
                    f"Could not connect to {context} at {self.base_url}: {exc}. "
                    "Check network access and GEMINI_API_KEY / GOOGLE_API_KEY."
                ),
                tokens=TokenUsage(),
            )
        if isinstance(exc, httpx.TimeoutException):
            return VlmResponse(
                content=(
                    f"Request to {context} timed out after {self.timeout}s. "
                    "Video analysis can take a while — raise timeout if needed."
                ),
                tokens=TokenUsage(),
            )
        if isinstance(exc, httpx.HTTPStatusError):
            body = exc.response.text[:200]
            if exc.response.status_code in (401, 403):
                return VlmResponse(
                    content=(
                        f"{context} rejected the API key (HTTP "
                        f"{exc.response.status_code}: {body}). Set GEMINI_API_KEY "
                        "or pass api_key in config."
                    ),
                    tokens=TokenUsage(),
                )
            return VlmResponse(
                content=f"{context} returned HTTP {exc.response.status_code}: {body}",
                tokens=TokenUsage(),
            )
        return VlmResponse(content=f"Unexpected error contacting {context}: {exc}", tokens=TokenUsage())


    def describe_image(
        self, image: np.ndarray, question: Optional[str] = None
    ) -> VlmResponse:
        """Describe one frame, or answer a question about it."""
        prompt = question or "Describe this image accurately and concisely."
        return self._generate([
            {"text": prompt},
            {"inline_data": {"mimeType": "image/jpeg", "data": encode_base64(image)}},
        ])

    def describe_frames(
        self,
        frames: list[tuple[float, np.ndarray]],
        question: Optional[str] = None,
    ) -> str:
        """Describe several frames with their timestamps."""
        if not frames:
            return ""
        header = (question or "Describe these video frames.")
        if len(frames) > 1:
            header += " Timestamps are listed in order."
        parts: list[dict] = [{"text": header}]
        for timestamp, image in frames:
            parts.append({"text": f"Frame at {timestamp:.1f} seconds:"})
            parts.append({"inline_data": {"mimeType": "image/jpeg", "data": encode_base64(image)}})
        return self._generate(parts).content

    def describe_video(
        self,
        video_uri: str,
        question: Optional[str] = None,
        mime_type: str = "video/mp4",
        start_offset: Optional[str] = None,
        end_offset: Optional[str] = None,
    ) -> VlmResponse:
        """Ask questions about a whole video without extracting frames.

        Args:
            video_uri: fileData fileUri — public HTTP(S) URL (small files only),
                a gs:// URI, or a single YouTube URL per request.
            question: Question or summarization prompt.
            mime_type: Media MIME type (e.g. video/mp4, video/webm).
            start_offset/end_offset: Optional ISO-8601 clip bounds, e.g. "0s"/"90s".

        Returns:
            VlmResponse with the answer and token usage.

        Note: local files must be uploaded first (Gemini Files API) to obtain a
        fileUri; this provider sends whatever URI the caller supplies.
        """
        prompt = question or "Describe what happens in this video, in order."
        file_data: dict = {"mimeType": mime_type, "fileUri": video_uri}
        parts: list[dict] = [{"text": prompt}]
        if start_offset or end_offset:
            clip: dict = {}
            if start_offset:
                clip["startOffset"] = start_offset
            if end_offset:
                clip["endOffset"] = end_offset
            parts.append({"fileData": file_data, "videoMetadata": clip})
        else:
            parts.append({"fileData": file_data})
        return self._generate(parts)

    def query_text(self, prompt: str) -> VlmResponse:
        """Send a text-only query (no media parts)."""
        return self._generate([{"text": prompt}])

    def check_health(self) -> bool:
        """Verify the model is reachable with the configured key."""
        if not self.api_key:
            return False
        try:
            response = self._client.get(f"{self.base_url}/models/{self.model_name}")
            return response.status_code == 200
        except Exception:
            return False

    def close(self) -> None:
        """Release the HTTP client."""
        self._client.close()


GoogleProvider = GeminiProvider
