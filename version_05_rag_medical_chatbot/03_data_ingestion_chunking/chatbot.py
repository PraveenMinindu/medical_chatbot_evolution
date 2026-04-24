"""Chatbot layer: safety checks, chunk retrieval, response assembly.

Day 3 change from Day 2:
    Responses are now text passages from a real document, not pre-written answers.
    The ChatResponse now exposes all_chunks, which is the context window
    that Day 4 will pass to the LLM for answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import FALLBACK_MESSAGE, RESTRICTED_KEYWORDS, SAFETY_MESSAGE
from .retrieval import KnowledgeRetriever, RetrievedChunk
from .utils import contains_restricted_keywords


@dataclass
class ChatResponse:
    """Complete response for one user message."""
    reply:         str
    response_type: str           # "safety" | "retrieval" | "fallback"
    best_score:    float | None
    best_topic:    str | None
    best_source:   str | None

    # All retrieved chunks -- Day 4 LLM context window
    all_chunks: list[RetrievedChunk] = field(default_factory=list)


class MedicalChatbot:
    """Orchestrates safety checking and chunk retrieval."""

    def __init__(self) -> None:
        self.retriever = KnowledgeRetriever()

    def get_response(self, user_input: str) -> ChatResponse:
        # Priority 1: safety
        if contains_restricted_keywords(user_input, RESTRICTED_KEYWORDS):
            return ChatResponse(
                reply         = SAFETY_MESSAGE,
                response_type = "safety",
                best_score    = None,
                best_topic    = None,
                best_source   = None,
                all_chunks    = [],
            )

        # Priority 2: chunk retrieval
        result = self.retriever.retrieve(user_input)

        # Priority 3: fallback
        if not result.has_results:
            return ChatResponse(
                reply         = FALLBACK_MESSAGE,
                response_type = "fallback",
                best_score    = result.best_score,
                best_topic    = None,
                best_source   = None,
                all_chunks    = [],
            )

        return ChatResponse(
            reply         = result.best_text,
            response_type = "retrieval",
            best_score    = result.best_score,
            best_topic    = result.best_topic,
            best_source   = result.best_source,
            all_chunks    = result.all_chunks,
        )
