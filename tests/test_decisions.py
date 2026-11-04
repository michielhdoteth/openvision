"""Tests for the typed decision-model tier."""
import json

import httpx
import numpy as np
import pytest

from core.errors import OpenVisionError
from providers.base import VlmProvider
from providers.decisions import DecisionAnswer, DecisionProvider
from providers.registry import ProviderRegistry
from typing import cast


def _provider(answer_payload: dict, captured: dict, **config) -> DecisionProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode())
        captured.setdefault("history", []).append(captured["body"])
        return httpx.Response(200, json=answer_payload)

    return DecisionProvider({**config, "transport": httpx.MockTransport(handler)})


def _frame(size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), 96, dtype=np.uint8)


def test_openrouter_request_shape_and_noul_parsing():
    captured: dict = {}
    payload = {
        "model": "openai/gpt-6-luna-decisions",
        "provider": "openrouter",
        "answers": {"answer": {"type": "noul", "noul": 0.91}},
        "usage": {"input_tokens": 812, "output_tokens": 0, "cost": 0.00008},
    }
    provider = _provider(payload, captured, api_key="test-key")
    try:
        result = provider.decide([_frame(), _frame()], "Is a person visible?", kind="noul")
    finally:
        provider.close()

    body = captured["body"]
    assert captured["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert captured["headers"]["authorization"] == "Bearer test-key"
    assert body["model"] == "openai/gpt-6-luna-decisions"
    assert body["questions"]["answer"]["type"] == "noul"
    assert body["questions"]["answer"]["instructions"] == "Is a person visible?"

    # state carries the prompt first, then one image part per frame
    assert body["state"][0] == "Is a person visible?"
    image_parts = [part for part in body["state"][1:] if isinstance(part, dict)]
    assert len(image_parts) == 2
    assert image_parts[0]["type"] == "image_url"
    assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    answer = result.first()
    assert answer.type == "noul"
    assert answer.probability == pytest.approx(0.91)
    assert answer.as_text() == "Yes (p=0.91)"
    assert result.tokens.total_tokens == 812
    assert result.cost == pytest.approx(0.00008)


def test_openai_preset_uses_decisions_api_wire_format():
    """OpenAI's /v1/decisions wants an input array plus a questions array."""
    captured: dict = {}
    payload = {"answers": [{"name": "answer", "type": "predicate", "probability": 0.22}]}
    provider = _provider(payload, captured, preset="openai", model="gpt-6-luna")
    try:
        result = provider.decide([_frame()], "Is the door open?", kind="noul")
    finally:
        provider.close()

    body = captured["body"]
    assert captured["url"] == "https://api.openai.com/v1/decisions"
    assert body["model"] == "gpt-6-luna"
    assert isinstance(body["input"], list) and "state" not in body
    question = body["questions"][0]
    assert question["name"] == "answer"
    assert question["type"] == "predicate"
    assert result.first().as_text() == "No (p=0.22)"


def test_openai_choice_question_uses_choices_array():
    captured: dict = {}
    payload = {"answers": [{"name": "action", "type": "choice", "choice": "billing",
                            "probabilities": {"billing": 0.9, "other": 0.1},
                            "confidence": 0.9}]}
    provider = _provider(payload, captured, preset="openai", model="gpt-6-luna")
    try:
        result = provider.decide(
            [_frame()], "Route this ticket.",
            kind="choice", criteria=["billing", "other"], question_id="action",
        )
    finally:
        provider.close()

    body = captured["body"]
    assert isinstance(body["questions"], list)
    assert body["questions"][0]["choices"] == [{"value": "billing"}, {"value": "other"}]
    assert result.first().value == "billing"


def test_openai_score_question_uses_levels_array():
    captured: dict = {}
    payload = {"answers": [{"name": "severity", "type": "score", "score": 1.1,
                            "probabilities": {0: 0.1, 1: 0.7, 2: 0.2},
                            "confidence": 0.55}]}
    provider = _provider(payload, captured, preset="openai", model="gpt-6-luna")
    try:
        result = provider.decide(
            [_frame()], "How bad is it?",
            kind="score", criteria=["low", "medium", "high"], question_id="severity",
        )
    finally:
        provider.close()

    body = captured["body"]
    assert body["questions"][0]["levels"] == [
        {"label": "low"}, {"label": "medium"}, {"label": "high"}
    ]
    assert result.first().value == pytest.approx(1.1)


def test_openai_refusal_answer_is_surfaced():
    captured: dict = {}
    payload = {"answers": [{"name": "answer", "type": "refusal",
                            "reason": "safety"}]}
    provider = _provider(payload, captured, preset="openai", model="gpt-6-luna")
    try:
        result = provider.decide([_frame()], "Is the door open?", kind="noul")
    finally:
        provider.close()

    assert result.first().type == "refusal"


def test_choice_question_declares_labels_and_renders_answer():
    captured: dict = {}
    payload = {
        "answers": {
            "answer": {
                "type": "choice",
                "choice": "walking",
                "probabilities": {"walking": 0.86, "sitting": 0.14},
                "confidence": 0.72,
            }
        }
    }
    provider = _provider(payload, captured)
    try:
        result = provider.decide(
            [_frame()],
            "What is the person doing?",
            kind="choice",
            criteria=["walking", "sitting"],
        )
    finally:
        provider.close()

    question = captured["body"]["questions"]["answer"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"walking", "sitting"}

    answer = result.first()
    assert answer.value == "walking"
    assert answer.as_text() == "walking (p=0.86, confidence 0.72)"


def test_score_question_uses_ordered_levels():
    captured: dict = {}
    payload = {
        "answers": {
            "answer": {
                "type": "score",
                "score": 2.0,
                "legend": {"0": "none", "1": "mild", "2": "severe"},
            }
        }
    }
    provider = _provider(payload, captured)
    try:
        result = provider.decide(
            [_frame()], "How severe is the damage?", kind="score", criteria=["none", "mild", "severe"]
        )
    finally:
        provider.close()

    assert captured["body"]["questions"]["answer"]["criteria"] == ["none", "mild", "severe"]
    assert result.first().as_text() == "severe (score=2.00)"


def test_choice_without_labels_is_rejected_before_network():
    provider = DecisionProvider({"transport": httpx.MockTransport(lambda r: httpx.Response(500))})
    try:
        with pytest.raises(OpenVisionError) as excinfo:
            provider.decide([_frame()], "What is happening?", kind="choice")
        assert excinfo.value.code == "DECISION_NEEDS_CHOICES"
    finally:
        provider.close()


def test_image_cap_is_per_model():
    assert DecisionProvider.max_images_for("cloudflare/clef") == 4
    assert DecisionProvider.max_images_for("cloudflare/clef-flash") == 4
    assert DecisionProvider.max_images_for("typesafe/jev-1.14") == 4
    assert DecisionProvider.max_images_for("openai/gpt-6-luna-decisions") == 128

    provider = DecisionProvider(
        {"model": "cloudflare/clef", "transport": httpx.MockTransport(lambda r: httpx.Response(200, json={}))}
    )
    try:
        with pytest.raises(OpenVisionError) as excinfo:
            provider.decide([_frame() for _ in range(5)], "Is a person visible?")
        assert excinfo.value.code == "DECISION_TOO_MANY_IMAGES"
    finally:
        provider.close()


def test_fit_image_enforces_pixel_window():
    small = DecisionProvider.fit_image(_frame(100))
    assert min(small.shape[:2]) >= 256

    large = DecisionProvider.fit_image(np.zeros((4000, 3000, 3), dtype=np.uint8))
    assert max(large.shape[:2]) <= 1568


def test_registry_alias_pins_model_and_kind():
    provider = ProviderRegistry.create({"provider": "clef"})
    try:
        decision_provider = cast(DecisionProvider, provider)
        assert decision_provider.kind == "decision"
        assert decision_provider.model_name == "cloudflare/clef"
        assert decision_provider.max_images == 4
    finally:
        provider.close()

    provider = ProviderRegistry.create({"provider": "openai-decisions", "model": "auto"})
    try:
        decision_provider = cast(DecisionProvider, provider)
        assert decision_provider.preset == "openai"
        assert decision_provider.model_name == "gpt-6-luna"
    finally:
        provider.close()


def test_chat_providers_advertise_chat_kind():
    assert VlmProvider.kind == "chat"
    assert "image" in VlmProvider.media


def test_ask_helpers_route_and_subsample():
    from cli.commands.ask import _evenly, _normalize_decision_kind

    assert _normalize_decision_kind(None, None, None) is None
    assert _normalize_decision_kind(None, "a,b", None) == "choice"
    assert _normalize_decision_kind(None, None, "low,high") == "score"
    assert _normalize_decision_kind("labels", None, None) == "choice"
    assert _normalize_decision_kind("bool", None, None) == "noul"

    frames = [{"timestamp": index} for index in range(10)]
    picked = _evenly(frames, 4)
    assert len(picked) == 4
    assert picked[0]["timestamp"] == 0
    assert picked[-1]["timestamp"] == 9
    assert _evenly(frames, 20) == frames

def test_run_decision_returns_ask_result(monkeypatch):
    from cli.commands import ask as ask_module

    captured: dict = {}
    payload = {
        "answers": {
            "answer": {
                "type": "choice",
                "choice": "sitting",
                "probabilities": {"sitting": 0.93, "walking": 0.07},
                "confidence": 0.9,
            }
        }
    }
    provider = _provider(payload, captured)
    monkeypatch.setattr(ask_module.ProviderRegistry, "create", lambda config: provider)

    frames = [
        {"image": _frame(), "timestamp": index, "timestamp_str": f"0:0{index}"}
        for index in range(6)
    ]
    result = ask_module._run_decision(
        frames,
        "What is the person doing?",
        "choice",
        "walking,sitting",
        None,
        0.6,
        {"models": {"decisions": {"preset": "openrouter"}}},
        None,
    )

    assert result.answer == "sitting (p=0.93, confidence 0.90)"
    assert result.confidence == "high"
    assert result.evidence[0]["frames"] == 6
    assert result.tokens.total_tokens == 0
    assert captured["body"]["questions"]["answer"]["criteria"] == {
        "walking": None,
        "sitting": None,
    }


def test_run_decision_respects_model_frame_cap(monkeypatch):
    from cli.commands import ask as ask_module

    captured: dict = {}
    payload = {"answers": {"answer": {"type": "noul", "noul": 0.7}}}
    provider = DecisionProvider(
        {"model": "cloudflare/clef", "transport": httpx.MockTransport(_ok(payload, captured))}
    )
    monkeypatch.setattr(ask_module.ProviderRegistry, "create", lambda config: provider)

    frames = [
        {"image": _frame(), "timestamp": index, "timestamp_str": f"0:0{index}"}
        for index in range(10)
    ]
    result = ask_module._run_decision(
        frames, "Is a person visible?", "noul", None, None, 0.6, {}, None
    )

    assert result.evidence[0]["frames"] == 4
    assert result.answer == "Yes (p=0.70)"
    assert len([p for p in captured["body"]["state"][1:] if isinstance(p, dict)]) == 4


def test_run_decision_rejects_chat_provider(monkeypatch):
    from cli.commands import ask as ask_module

    class ChatStub:
        kind = "chat"
        model_name = "gemini"

        def close(self):
            pass

    monkeypatch.setattr(ask_module.ProviderRegistry, "create", lambda config: ChatStub())

    with pytest.raises(OpenVisionError) as excinfo:
        ask_module._run_decision(
            [{"image": _frame(), "timestamp": 0, "timestamp_str": "0:00"}],
            "Is a person visible?",
            "noul",
            None,
            None,
            0.6,
            {},
            "gemini",
        )
    assert excinfo.value.code == "DECISION_WRONG_PROVIDER"


def _ok(payload, captured):
    def handler(request):
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=payload)

    return handler

def test_zen_preset_is_text_only():
    captured: dict = {}
    payload = {
        "model": "jev-1.13-free",
        "answers": {"answer": {"type": "noul", "noul": 0.97}},
        "usage": {"input_tokens": 315, "output_tokens": 20, "cost": 0},
    }
    provider = _provider(payload, captured, preset="zen")
    try:
        assert provider.base_url == "https://opencode.ai/zen/v1/systemone"
        assert provider.supports_images is False
        assert provider.media == ("text",)
        assert provider.max_images == 0

        result = provider.decide(
            instructions="Is this support request urgent?",
            state_text="Checkout has been failing for every customer for the last hour.",
            kind="noul",
        )
        body = captured["body"]
        assert body["state"] == [
            "Checkout has been failing for every customer for the last hour.",
            "Is this support request urgent?",
        ]
        assert result.first().as_text() == "Yes (p=0.97)"
        assert result.cost == 0
    finally:
        provider.close()


def test_zen_rejects_frames_instead_of_billing_them_as_text():
    provider = DecisionProvider(
        {"preset": "zen", "transport": httpx.MockTransport(lambda r: httpx.Response(200, json={}))}
    )
    try:
        with pytest.raises(OpenVisionError) as excinfo:
            provider.decide([_frame()], "Is a person visible?", kind="noul")
        assert excinfo.value.code == "DECISION_TEXT_ONLY"
        assert excinfo.value.suggested_retry["provider"] == "openrouter"
    finally:
        provider.close()


def test_zen_preset_inferred_from_base_url():
    provider = DecisionProvider(
        {
            "base_url": "https://opencode.ai/zen/v1/systemone",
            "transport": httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        }
    )
    try:
        assert provider.preset == "zen"
        assert provider.supports_images is False
    finally:
        provider.close()

    provider = ProviderRegistry.create({"provider": "jev-zen"})
    try:
        decision_provider = cast(DecisionProvider, provider)
        assert decision_provider.preset == "zen"
        assert decision_provider.model_name == "jev-1.13-free"
    finally:
        provider.close()
