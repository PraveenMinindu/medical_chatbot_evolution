from __future__ import annotations

import random
import re
from typing import Iterable

from .config import (
    CONFIDENCE_THRESHOLD,
    DISCLAIMER_MESSAGE,
    FALLBACK_MESSAGE,
    INTENTS_PATH,
    SAFE_SCOPE_MESSAGE,
    SUPPORTED_TOPICS,
    URGENT_MESSAGE,
)
from .predict import PredictionResult, predict_intent
from .utils import build_response_lookup, choose_response, load_intents


# FIX: Split 'treat' and 'treatment' into separate entries so whole-word
# boundary matching (\b) catches each correctly as a standalone word.
# 'treat' hits: "how to treat a plant"   → blocked (correct)
# 'treatment' hits: "treatment plant"    → also blocked (intended for medical safety)
# But "how to treat a plant" is a false positive — acceptable trade-off at this
# project stage. A+++ and A++++ will use semantic understanding instead.
RISKY_KEYWORDS = {
    "medicine",
    "tablet",
    "tablets",
    "drug",
    "drugs",
    "antibiotic",
    "antibiotics",
    "prescribe",
    "prescription",
    "diagnose",
    "diagnosis",
    "treatment",
    "treat",
    "cure",
    "emergency",
    "medicate",
}

URGENT_KEYWORDS = {
    "severe",
    "urgent",
    "emergency",
    "unconscious",
    "chest pain",
    "cant breathe",
    "cannot breathe",
    "bleeding",
    "am i dying",
    "dying",
}


class MedicalChatbot:
    def __init__(self, seed: int | None = None) -> None:
        intents = load_intents(INTENTS_PATH)
        self.response_lookup = build_response_lookup(intents)
        self.rng = random.Random(seed)

    def _contains_any_phrase(self, text: str, phrases: Iterable[str]) -> bool:
        # FIX: Use whole-word boundary matching (\b) instead of plain substring search.
        #
        # Old code: any(phrase in text for phrase in phrases)
        # Bug example: "treat" matched inside "treatment plant explained"
        #              because "treat" IS a substring of "treatment".
        #
        # New code: re.search(r"\btreat\b", text)
        # Result:   "treat" does NOT match inside "treatment" because
        #           \b checks for a word boundary — the 'm' in "treatment"
        #           is a word character, so \b does not fire after "treat".
        #
        # This is the same fix applied in Version A and A+.
        return any(
            re.search(r"\b" + re.escape(phrase) + r"\b", text)
            for phrase in phrases
        )

    def _is_urgent(self, cleaned_text: str) -> bool:
        return self._contains_any_phrase(cleaned_text, URGENT_KEYWORDS)

    def _is_risky(self, cleaned_text: str) -> bool:
        return self._contains_any_phrase(cleaned_text, RISKY_KEYWORDS)

    def _build_topic_hint(self) -> str:
        topic_text = ", ".join(SUPPORTED_TOPICS[:-1]) + f", and {SUPPORTED_TOPICS[-1]}"
        return f"You can ask me about {topic_text}."

    def generate_reply(self, user_text: str) -> dict[str, str | float]:
        prediction: PredictionResult = predict_intent(user_text)

        if self._is_urgent(prediction.cleaned_text):
            return {
                "reply": URGENT_MESSAGE,
                "predicted_intent": "urgent_support",
                "confidence": prediction.confidence,
            }

        if prediction.tag == "out_of_scope" or self._is_risky(prediction.cleaned_text):
            return {
                "reply": SAFE_SCOPE_MESSAGE,
                "predicted_intent": "out_of_scope",
                "confidence": prediction.confidence,
            }

        if prediction.confidence < CONFIDENCE_THRESHOLD:
            return {
                "reply": f"{FALLBACK_MESSAGE} {self._build_topic_hint()}",
                "predicted_intent": "fallback",
                "confidence": prediction.confidence,
            }

        responses = self.response_lookup.get(prediction.tag, [])
        reply = choose_response(responses, self.rng) or FALLBACK_MESSAGE

        if prediction.tag in {"greeting", "help", "bot_identity"}:
            reply = f"{reply} {DISCLAIMER_MESSAGE}"

        return {
            "reply": reply,
            "predicted_intent": prediction.tag,
            "confidence": prediction.confidence,
        }
