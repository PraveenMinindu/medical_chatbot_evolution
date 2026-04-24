"""Chatbot layer: safety checks, retrieval, response assembly.

Day 1: returned one answer with one score.
Day 2: returns best answer + all top_k retrieved chunks as sources.

The top_chunks list is what a future LLM (Day 3+) will receive as its
context. For now we display them as "retrieved sources" so the user can
see how the retrieval system works internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import FALLBACK_MESSAGE, RESTRICTED_KEYWORDS, SAFETY_MESSAGE
from .retrieval import KnowledgeRetriever, RetrievalResponse, RetrievedChunk
from .utils import contains_restricted_keywords


@dataclass
class ChatResponse:
    """Complete response for one user message."""
    reply:          str
    response_type:  str             # "safety" | "retrieval" | "fallback"
    best_score:     float | None    # similarity score of the top result
    best_topic:     str | None
    best_source_id: str | None

    # All retrieved chunks above threshold -- what an LLM would see as context
    sources: list[RetrievedChunk] = field(default_factory=list)


class MedicalChatbot:
    """Orchestrates safety checking, top-k retrieval, and response assembly.

    Response priority:
        1. Safety check  -- restricted keyword detected.
        2. Retrieval     -- FAISS top-k search returns valid results.
        3. Fallback      -- all scores below threshold.
    """

    def __init__(self, rebuild: bool = False) -> None:
        self.retriever = KnowledgeRetriever(rebuild=rebuild)

    def get_response(self, user_input: str) -> ChatResponse:
        """Process user input and return a structured ChatResponse.

        Args:
            user_input: Raw text from the user.

        Returns:
            ChatResponse with reply, sources, scores, and response type.
        """
        # Priority 1: safety check
        if contains_restricted_keywords(user_input, RESTRICTED_KEYWORDS):
            return ChatResponse(
                reply          = SAFETY_MESSAGE,
                response_type  = "safety",
                best_score     = None,
                best_topic     = None,
                best_source_id = None,
                sources        = [],
            )

        # Priority 2: top-k retrieval from FAISS
        result: RetrievalResponse = self.retriever.retrieve(user_input)

        # Priority 3: fallback if no results above threshold
        if not result.has_results:
            return ChatResponse(
                reply          = FALLBACK_MESSAGE,
                response_type  = "fallback",
                best_score     = result.best_score,
                best_topic     = None,
                best_source_id = None,
                sources        = [],
            )

        return ChatResponse(
            reply          = result.best_answer,
            response_type  = "retrieval",
            best_score     = result.best_score,
            best_topic     = result.best_topic,
            best_source_id = result.best_source_id,
            sources        = result.top_chunks,
        )
