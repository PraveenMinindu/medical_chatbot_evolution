"""Safety guardrails -- the first line of defence before retrieval or generation.

WHY A DEDICATED SAFETY MODULE:
    In Day 3 the safety check was a single function in utils.py.
    As the system grows, safety logic becomes more complex:
      - Restricted keyword blocking (diagnosis, prescriptions, etc.)
      - Urgent phrase detection (chest pain, cannot breathe, etc.)
      - Output scanning to catch LLM responses that slip past input checks

    Keeping all safety logic in one module makes it easy to audit,
    test, and extend independently of the rest of the pipeline.

HOW THIS FITS INTO RAG:
    Safety runs BEFORE retrieval and BEFORE generation.
    No embedding computation, no FAISS search, no LLM call is made
    for blocked queries. This is both safer and more efficient.

    The output scanner runs AFTER generation as a secondary check.
    If the LLM somehow produces a response containing restricted content,
    the scanner catches and replaces it.
"""

from __future__ import annotations

from app.config import RESTRICTED_KEYWORDS, SAFETY_MESSAGE, URGENT_MESSAGE, URGENT_PHRASES
from app.utils.text import contains_whole_word


# Words that should never appear in a generated answer.
# These are checked in the output scanner after LLM generation.
BLOCKED_OUTPUT_PHRASES = {
    "you should take",
    "i recommend taking",
    "the dosage is",
    "take this medicine",
    "prescribed for",
    "you need antibiotics",
    "go to the emergency",
}


class GuardrailResult:
    """Result of running safety checks on a piece of text."""

    def __init__(
        self,
        blocked: bool,
        reason: str,
        safe_response: str | None = None,
    ) -> None:
        self.blocked       = blocked
        self.reason        = reason
        self.safe_response = safe_response   # the message to return if blocked


def check_input(user_input: str) -> GuardrailResult:
    """Run all input-level safety checks on raw user text.

    Checks in order:
        1. Urgent phrases -- requires immediate attention, different message.
        2. Restricted keywords -- diagnosis, prescriptions, etc.

    Args:
        user_input: Raw text from the user.

    Returns:
        GuardrailResult with blocked=True if any check fails.
    """
    # Check 1: urgent phrases (before keyword check -- different message)
    if contains_whole_word(user_input, URGENT_PHRASES):
        return GuardrailResult(
            blocked       = True,
            reason        = "urgent_phrase",
            safe_response = URGENT_MESSAGE,
        )

    # Check 2: restricted keywords
    if contains_whole_word(user_input, RESTRICTED_KEYWORDS):
        return GuardrailResult(
            blocked       = True,
            reason        = "restricted_keyword",
            safe_response = SAFETY_MESSAGE,
        )

    return GuardrailResult(blocked=False, reason="ok")


def scan_output(generated_text: str) -> GuardrailResult:
    """Scan LLM-generated text for any policy-violating content.

    This is a secondary safety check that runs after generation.
    It is not a substitute for good prompt engineering -- it is
    a backstop for edge cases.

    Args:
        generated_text: Text produced by the LLM.

    Returns:
        GuardrailResult with blocked=True if the output is unsafe.
    """
    lower = generated_text.lower()
    for phrase in BLOCKED_OUTPUT_PHRASES:
        if phrase in lower:
            return GuardrailResult(
                blocked       = True,
                reason        = f"unsafe_output_phrase: '{phrase}'",
                safe_response = SAFETY_MESSAGE,
            )

    return GuardrailResult(blocked=False, reason="ok")
