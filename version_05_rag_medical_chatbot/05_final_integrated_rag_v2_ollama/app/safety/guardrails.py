"""Input and output safety guardrails."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import (
    BLOCKED_OUTPUT_PHRASES,
    RESTRICTED_KEYWORDS,
    SAFETY_MESSAGE,
    URGENT_MESSAGE,
    URGENT_PHRASES,
)
from app.utils.text import contains_whole_word


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str
    safe_response: str | None = None


def check_input(user_input: str) -> GuardrailResult:
    if contains_whole_word(user_input, URGENT_PHRASES):
        return GuardrailResult(True, "urgent_phrase", URGENT_MESSAGE)
    if contains_whole_word(user_input, RESTRICTED_KEYWORDS):
        return GuardrailResult(True, "restricted_keyword", SAFETY_MESSAGE)
    return GuardrailResult(False, "ok")


def scan_output(generated_text: str) -> GuardrailResult:
    lower = generated_text.lower()
    for phrase in BLOCKED_OUTPUT_PHRASES:
        if phrase in lower:
            return GuardrailResult(True, f"unsafe_output:{phrase}", SAFETY_MESSAGE)
    return GuardrailResult(False, "ok")

