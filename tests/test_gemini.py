"""Tests for the Gemini native-API provider (no network calls)."""
import httpx
import numpy as np
import pytest

from providers.gemini import GeminiProvider, DEFAULT_BASE_URL, DEFAULT_MODEL


def _provider(monkeypatch, **config):
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    return GeminiProvider(config)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "boom", request=httpx.Request("POST", "http://x"), response=self._as_response()
            )

    def _as_response(self):
        return httpx.Response(self.status_code, json=self._payload)

    def json(self):
        return self._payload


def test_defaults(monkeypatch):
    provider = _provider(monkeypatch)
    try:
        assert provider.base_url == DEFAULT_BASE_URL
        assert provider.model_name == DEFAULT_MODEL
        assert provider.PROVIDER_NAME == "gemini"
        assert provider._client.headers["x-goog-api-key"] == "unit-test-key"
    finally:
        provider.close()


def test_auto_model_sentinel_falls_back(monkeypatch):
    provider = _provider(monkeypatch, model="auto")
    try:
        assert provider.model_name == DEFAULT_MODEL
    finally:
        provider.close()


def test_google_api_key_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    provider = GeminiProvider({})
    try:
        assert provider.api_key == "google-key"
    finally:
        provider.close()


def test_base_url_override_and_trailing_slash(monkeypatch):
    provider = _provider(monkeypatch, base_url="https://proxy.example.com/v1beta/")
    try:
        assert provider.base_url == "https://proxy.example.com/v1beta"
    finally:
        provider.close()


def test_generate_posts_text_and_parses_response(monkeypatch):
    provider = _provider(monkeypatch)
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse({
            "candidates": [{"content": {"parts": [{"text": "a dial turns slowly"}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 34,
                              "thoughtsTokenCount": 5, "totalTokenCount": 51},
        })

    monkeypatch.setattr(provider._client, "post", fake_post)
    response = provider.query_text("What happens?")
    try:
        assert response.content == "a dial turns slowly"
        assert response.tokens.prompt_tokens == 12
        assert response.tokens.reasoning_tokens == 5
        assert captured["url"].endswith("/models/gemini-3.1-flash-lite:generateContent")
        assert captured["payload"]["contents"][0]["parts"] == [{"text": "What happens?"}]
        assert captured["payload"]["generationConfig"]["temperature"] == 0.1
    finally:
        provider.close()


def test_describe_image_sends_inline_jpeg(monkeypatch):
    provider = _provider(monkeypatch)
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["payload"] = json
        return _FakeResponse({"candidates": [{"content": {"parts": [{"text": "a red dial"}]}}]})

    monkeypatch.setattr(provider._client, "post", fake_post)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    provider.describe_image(image, "what color is the dial?")
    try:
        parts = captured["payload"]["contents"][0]["parts"]
        assert parts[0]["text"] == "what color is the dial?"
        assert parts[1]["inline_data"]["mimeType"] == "image/jpeg"
        assert parts[1]["inline_data"]["data"]
    finally:
        provider.close()


def test_describe_video_sends_file_data(monkeypatch):
    provider = _provider(monkeypatch)
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["payload"] = json
        return _FakeResponse({"candidates": [{"content": {"parts": [{"text": "summary"}]}}]})

    monkeypatch.setattr(provider._client, "post", fake_post)
    provider.describe_video(
        "https://example.com/clip.mp4", "Summarize", mime_type="video/mp4",
        start_offset="10s", end_offset="70s",
    )
    try:
        media_part = captured["payload"]["contents"][0]["parts"][1]
        assert media_part["fileData"]["fileUri"] == "https://example.com/clip.mp4"
        assert media_part["fileData"]["mimeType"] == "video/mp4"
        assert media_part["videoMetadata"] == {"startOffset": "10s", "endOffset": "70s"}
    finally:
        provider.close()


def test_describe_frames_timestamps_in_text(monkeypatch):
    provider = _provider(monkeypatch)
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["payload"] = json
        return _FakeResponse({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    monkeypatch.setattr(provider._client, "post", fake_post)
    frames = [(1.0, np.zeros((4, 4, 3), dtype=np.uint8)), (31.5, np.zeros((4, 4, 3), dtype=np.uint8))]
    assert provider.describe_frames(frames, "any motion?") == "ok"
    texts = [p.get("text", "") for p in captured["payload"]["contents"][0]["parts"]]
    try:
        assert any("Frame at 1.0 seconds" in t for t in texts)
        assert any("Frame at 31.5 seconds" in t for t in texts)
    finally:
        provider.close()


def test_http_error_becomes_readable_response(monkeypatch):
    provider = _provider(monkeypatch)

    def fake_post(url, json=None, **kwargs):
        return _FakeResponse({"error": "no key"}, status_code=403)

    monkeypatch.setattr(provider._client, "post", fake_post)
    response = provider.query_text("hi")
    try:
        assert "rejected the API key" in response.content
    finally:
        provider.close()


def test_check_health_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiProvider({})
    try:
        assert provider.check_health() is False
    finally:
        provider.close()


def test_check_health_reachable(monkeypatch):
    provider = _provider(monkeypatch)

    def fake_get(url, **kwargs):
        return _FakeResponse({})

    monkeypatch.setattr(provider._client, "get", fake_get)
    try:
        assert provider.check_health() is True
        assert provider.check_health.__doc__
    finally:
        provider.close()
