"""High-level chatbot: safety checks, retrieval, and response generation."""

from __future__ import annotations

from dataclasses import dataclass

from config import(
    FALLBACK_MESSAGE,
    RESTRICTED_KEYWORDS,
    SAFETY_MESSAGE,
)
from retrieval import KnowledgeRetriever, RetrievalResult
from utils import contains_restricted_keywords


@dataclass
class ChatResponse:
    """The complete response returned for each user message."""
    reply: str
    source_id: str | None
    topic: str | None
    similarity_score: float | None
    response_type: str  # "safety" | "retrieval" | "fallback"


class MedicalChatbot:
    """Orchestrates safety checking and semantic retrieval.

    Response priority:
        1. Safety check  -- if restricted keywords found, return safety message.
        2. Retrieval     -- embed query and find best matching KB entry.
        3. Threshold     -- if similarity too low, return fallback message.
    """

    def __init__(self) -> None:
        self.retriever = KnowledgeRetriever()

    def get_response(self, user_input: str) -> ChatResponse:
        """Process user input and return a complete ChatResponse.

        Args:
            user_input: Raw text from the user.

        Returns:
            ChatResponse with reply, metadata, and response type.
        """
        # Priority 1: safety check before any retrieval
        if contains_restricted_keywords(user_input, RESTRICTED_KEYWORDS):
            return ChatResponse(
                reply=SAFETY_MESSAGE,
                source_id=None,
                topic=None,
                similarity_score=None,
                response_type="safety",
            )

        # Priority 2: semantic retrieval
        result: RetrievalResult = self.retriever.get_best_match(user_input)

        # Priority 3: threshold check
        if not result.above_threshold:
            return ChatResponse(
                reply=FALLBACK_MESSAGE,
                source_id=None,
                topic=None,
                similarity_score=result.similarity_score,
                response_type="fallback",
            )

        return ChatResponse(
            reply=result.answer,
            source_id=result.source_id,
            topic=result.topic,
            similarity_score=result.similarity_score,
            response_type="retrieval",
        )
