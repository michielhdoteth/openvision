"""Typed decision models: a ``state`` plus typed questions in, probabilities out.

The frames travel inside ``state``, so a free-form question still works once its
answer space is declared (``choice`` labels, ordered ``score`` levels, ``noul``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import httpx
import numpy as np

from core.errors import OpenVisionError
from core.image import encode_base64
from providers.base import TokenUsage

PRESETS: dict[str, dict[str, Any]] = {
    # OpenRouter's alpha decisions endpoint shape is unverified: keep the
    # System One body it was documented with until confirmed live.
    "openrouter": {
        "base_url": "https://openrouter.ai/api/alpha/decisions",
        "health_url": "https://openrouter.ai/api/v1/models",
        "api_key_envs": ("OPENROUTER_API_KEY",),
        "default_model": "openai/gpt-6-luna-decisions",
        "predicate_type": "noul",
        "supports_images": True,
        "wire": "systemone",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/decisions",
        "health_url": "https://api.openai.com/v1/models",
        "api_key_envs": ("OPENAI_API_KEY",),
        "default_model": "gpt-6-luna",
        "predicate_type": "predicate",
        "supports_images": True,
        "wire": "openai",
    },
    # Text-only: Zen tokenizes image parts as text instead of decoding them.
    # Zen serves the Jev System One wire format, which differs from OpenAI's
    # Decisions API in body and response shape.
    "zen": {
        "base_url": "https://opencode.ai/zen/v1/systemone",
        "health_url": "https://opencode.ai/zen/v1/models",
        "api_key_envs": ("OPENCODE_API_KEY",),
        "default_model": "jev-1.13-free",
        "predicate_type": "noul",
        "supports_images": False,
        "wire": "systemone",
    },
}

_MAX_IMAGES = (
    ("luna", 128),
    ("clef", 4),
    ("jev", 4),
    ("mercury", 16),
    ("mammoth", 16),
    ("elephant", 16),
)
_MAX_SIDE = 1568
_MIN_SIDE = 256
_JPEG_QUALITY = 85

@dataclass
class DecisionAnswer:
    type: str
    value: Optional[float | str] = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    legend: Optional[dict[str, str]] = None

    @property
    def probability(self) -> Optional[float]:
        """Truth probability for a noul/predicate answer."""
        if self.type == "noul" and isinstance(self.value, (int, float)):
            return float(self.value)
        return None

    def agreement(self) -> float:
        """How decisive the answer is, on a 0-1 scale."""
        if self.type == "noul":
            p = self.probability
            if p is None:
                return 0.0
            return max(p, 1.0 - p)
        if self.confidence is not None:
            return float(self.confidence)
        if self.probabilities:
            return float(max(self.probabilities.values()))
        return 0.0

    def as_text(self, threshold: float = 0.5) -> str:
        """Render the answer as one line of text for AskResult."""
        if self.type == "noul":
            p = self.probability or 0.0
            verdict = "Yes" if p >= threshold else "No"
            return f"{verdict} (p={p:.2f})"
        if self.type == "choice":
            p = self.probabilities.get(str(self.value), 0.0)
            if self.confidence is not None:
                return f"{self.value} (p={p:.2f}, confidence {self.confidence:.2f})"
            return f"{self.value} (p={p:.2f})"
        if self.type == "score":
            score = float(self.value or 0.0)
            levels = len(self.legend or {}) or None
            label = None
            if self.legend:
                label = self.legend.get(str(int(round(score))))
            where = f"level {round(score)}"
            if levels:
                where += f" of {levels - 1}"
            return f"{label or where} (score={score:.2f})"
        return str(self.value)

@dataclass
class DecisionResult:

    answers: dict[str, DecisionAnswer]
    tokens: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""
    provider: str = ""
    cost: Optional[float] = None

    def first(self, question_id: str = "answer") -> DecisionAnswer:
        if question_id in self.answers:
            return self.answers[question_id]
        if self.answers:
            return next(iter(self.answers.values()))
        raise OpenVisionError("DECISION_EMPTY", "Decision model returned no answers.")

class DecisionProvider:
    """Client for typed decision models.

    Keys: preset, base_url, model, api_key, api_key_env, timeout, max_images,
    transport (tests).
    """

    kind = "decision"
    media = ("text", "image")

    def __init__(self, config: dict):
        self.config = config
        base_url = str(config.get("base_url") or "")
        if config.get("preset"):
            preset = config["preset"]
        elif "opencode.ai" in base_url:
            preset = "zen"
        elif "api.openai.com" in base_url:
            preset = "openai"
        else:
            preset = "openrouter"
        self.preset = preset if preset in PRESETS else "openrouter"
        spec = PRESETS[self.preset]

        self.model_name = config.get("model") or spec["default_model"]
        if self.model_name == "auto":
            self.model_name = spec["default_model"]
        self.base_url = (base_url or spec["base_url"]).rstrip("/")
        self.timeout = config.get("timeout", 90)
        self.predicate_type = spec["predicate_type"]
        self.supports_images = bool(spec.get("supports_images", True))
        # "openai" (OpenAI Decisions API) and "systemone" (Jev) share the
        # question vocabulary but differ in body and response shape.
        self.wire = spec.get("wire", "systemone")
        self.media = ("text", "image") if self.supports_images else ("text",)
        if not self.supports_images:
            self.max_images = 0
        else:
            self.max_images = int(
                config.get("max_images") or self.max_images_for(self.model_name)
            )

        api_key = config.get("api_key")
        key_envs = (config.get("api_key_env"),) if config.get("api_key_env") else spec["api_key_envs"]
        if not api_key:
            for env in key_envs:
                if env and os.environ.get(env):
                    api_key = os.environ[env]
                    break
        self._api_key = api_key

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        client_kwargs: dict[str, Any] = {"timeout": self.timeout, "headers": headers}
        if config.get("transport") is not None:
            client_kwargs["transport"] = config["transport"]
        self._client = httpx.Client(**client_kwargs)

    @staticmethod
    def max_images_for(model: str) -> int:
        lowered = (model or "").lower()
        for fragment, cap in _MAX_IMAGES:
            if fragment in lowered:
                return cap
        return 8

    @classmethod
    def fit_image(cls, image: np.ndarray, jpeg: bool = True) -> np.ndarray:
        """Resize into the size window every decisions backend accepts."""
        h, w = image.shape[:2]
        if max(h, w) > _MAX_SIDE:
            scale = _MAX_SIDE / max(h, w)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
            h, w = image.shape[:2]
        if min(h, w) < _MIN_SIDE:
            scale = _MIN_SIDE / max(min(h, w), 1)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
            )
        return image

    def build_question(
        self,
        kind: str,
        instructions: str,
        criteria: Any = None,
    ) -> dict:
        """Build one typed question in this preset's wire format."""
        kind = kind.lower()
        if kind in ("noul", "yes/no", "bool", "predicate"):
            kind = "noul"
        if kind not in ("noul", "choice", "score"):
            raise OpenVisionError(
                "DECISION_BAD_KIND",
                f"Unknown decision type '{kind}'. Use noul, choice, or score.",
            )

        question: dict[str, Any] = {"type": self.predicate_type if kind == "noul" else kind}
        question["instructions"] = instructions

        if kind == "choice":
            if isinstance(criteria, dict):
                # Already criteria-shaped (System One); OpenAI adds a
                # description per option when the values carry one.
                if self.wire == "openai":
                    question["choices"] = [
                        {"value": label,
                         **({"description": desc} if desc else {})}
                        for label, desc in criteria.items()
                    ]
                else:
                    question["criteria"] = criteria
            elif isinstance(criteria, Sequence) and not isinstance(criteria, str):
                labels = [str(item) for item in criteria]
                if len(labels) < 2:
                    raise OpenVisionError(
                        "DECISION_NEEDS_CHOICES",
                        "A choice question needs at least two candidate labels.",
                        {"flag": "--choices"},
                    )
                if self.wire == "openai":
                    question["choices"] = [{"value": label} for label in labels]
                else:
                    question["criteria"] = {label: None for label in labels}
            else:
                raise OpenVisionError(
                    "DECISION_NEEDS_CHOICES",
                    "A choice question needs candidate labels.",
                    {"flag": "--choices"},
                )
        elif kind == "score":
            levels = list(criteria or [])
            if len(levels) < 2:
                raise OpenVisionError(
                    "DECISION_NEEDS_LEVELS",
                    "A score question needs at least two ordered levels.",
                    {"flag": "--levels"},
                )
            if self.wire == "openai":
                question["levels"] = [{"label": str(level)} for level in levels]
            else:
                question["criteria"] = [str(level) for level in levels]

        return question

    def decide(
        self,
        images: Optional[Sequence[np.ndarray]] = None,
        instructions: str = "",
        kind: str = "noul",
        criteria: Any = None,
        question_id: str = "answer",
        state_text: Optional[str] = None,
    ) -> DecisionResult:
        """Ask one typed question about a shared set of frames or a text state."""
        images = list(images or [])

        if images and not self.supports_images:
            raise OpenVisionError(
                "DECISION_TEXT_ONLY",
                f"{self.preset} decisions are text-only: images would be billed as "
                "text and ignored. Use a vision-capable decision route "
                "(openrouter, or Jev's own API) for frames.",
                {"preset": self.preset, "provider": "openrouter"},
            )
        if not images and not state_text:
            raise OpenVisionError(
                "DECISION_NO_INPUT",
                "A decision request needs at least one image or a text state.",
            )
        if len(images) > self.max_images:
            raise OpenVisionError(
                "DECISION_TOO_MANY_IMAGES",
                f"{self.model_name} accepts at most {self.max_images} images per request "
                f"but {len(images)} were selected.",
                {"max_images": self.max_images},
            )

        state: list[Any] = []
        if state_text:
            state.append(state_text)
        if instructions:
            state.append(instructions)
        for image in images:
            state.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_base64(self.fit_image(image), quality=_JPEG_QUALITY)
                    },
                }
            )

        question = self.build_question(kind, instructions, criteria)
        if self.wire == "openai":
            payload = {
                "model": self.model_name,
                "input": state,
                "questions": [{"name": question_id, **question}],
            }
        else:
            payload = {
                "model": self.model_name,
                "state": state,
                "questions": {question_id: question},
            }

        try:
            response = self._client.post(self.base_url, json=payload)
        except httpx.TimeoutException as exc:
            raise OpenVisionError(
                "DECISION_TIMEOUT",
                f"Decision request to {self.base_url} timed out after {self.timeout}s.",
                {"timeout": self.timeout},
            ) from exc
        except httpx.ConnectError as exc:
            raise OpenVisionError(
                "DECISION_UNREACHABLE",
                f"Could not reach {self.base_url}: {exc}",
            ) from exc

        if response.status_code == 401:
            raise OpenVisionError(
                "DECISION_AUTH_FAILED",
                f"{self.preset} rejected the API key. Set "
                f"{'/'.join(PRESETS[self.preset]['api_key_envs'])}.",
            )
        if response.status_code == 402:
            raise OpenVisionError(
                "DECISION_OUT_OF_CREDITS",
                "The decision provider reported an empty credit balance.",
            )
        if response.status_code == 413:
            raise OpenVisionError(
                "DECISION_INPUT_TOO_LARGE",
                "The request exceeded the provider's size limit. "
                "Use fewer frames or smaller --max-frames.",
                {"max_images": self.max_images},
            )
        if response.status_code == 429:
            raise OpenVisionError(
                "DECISION_RATE_LIMITED",
                "Decision provider rate limit hit. Retry shortly.",
                {"retry_after_seconds": 2},
            )
        if response.status_code >= 400:
            raise OpenVisionError(
                "DECISION_REQUEST_FAILED",
                f"{self.base_url} returned HTTP {response.status_code}: "
                f"{response.text[:300]}",
            )

        data = response.json()
        raw_answers = data.get("answers") or {}
        answers: dict[str, DecisionAnswer] = {}
        if isinstance(raw_answers, dict):
            # System One: answers keyed by question id.
            for key, raw in raw_answers.items():
                answers[key] = self._parse_answer(raw)
        elif isinstance(raw_answers, list):
            # OpenAI Decisions: an array of named answers, one per question.
            for index, raw in enumerate(raw_answers):
                key = str(raw.get("name") or index)
                answers[key] = self._parse_answer(raw)

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        tokens = TokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

        return DecisionResult(
            answers=answers,
            tokens=tokens,
            model=str(data.get("model") or self.model_name),
            provider=str(data.get("provider") or self.preset),
            cost=usage.get("cost"),
        )

    @staticmethod
    def _parse_answer(raw: dict) -> DecisionAnswer:
        kind = raw.get("type") or ""
        if kind == "refusal":
            # OpenAI's Decisions API refuses rather than guessing.
            return DecisionAnswer(
                type="refusal",
                value=raw.get("reason") or raw.get("message") or "refused",
            )
        if kind == "choice":
            return DecisionAnswer(
                type="choice",
                value=raw.get("choice"),
                probabilities={
                    str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()
                },
                confidence=raw.get("confidence"),
            )
        if kind == "score":
            return DecisionAnswer(
                type="score",
                value=raw.get("score"),
                probabilities={
                    str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()
                },
                confidence=raw.get("confidence"),
                legend={str(k): str(v) for k, v in (raw.get("legend") or {}).items()},
            )
        probability = raw.get("noul")
        if probability is None:
            probability = raw.get("predicate")
        if probability is None:
            probability = raw.get("probability")
        return DecisionAnswer(type="noul", value=probability)

    def check_health(self) -> bool:
        if not self._api_key:
            return False
        health_url = PRESETS[self.preset]["health_url"]
        try:
            response = self._client.get(health_url, timeout=min(self.timeout, 10))
            return response.status_code < 400
        except Exception:
            return False

    def close(self) -> None:
        self._client.close()
