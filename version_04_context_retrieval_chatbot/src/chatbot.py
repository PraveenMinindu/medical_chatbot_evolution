"""High-level chatbot logic with retrieval and safety checks."""

from __future__ import annotations

from config import (
    FALLBACK_MESSAGE,
    RESTRICTED_KEYWORDS,
    SAFETY_MESSAGE,
    SIMILARITY_THRESHOLD,
)
from retrieval import KnowledgeRetriever
from utils import contains_restricted_keywords


class MedicalChatbot:
    """Educational medical chatbot powered by information retrieval."""

    def __init__(self) -> None:
        self.retriever = KnowledgeRetriever()

    def get_response(self, user_input: str) -> tuple[str, float]:
        """Return a safe response and similarity score."""
        if contains_restricted_keywords(user_input, RESTRICTED_KEYWORDS):
            return SAFETY_MESSAGE, 1.0

        best_answer, similarity_score = self.retriever.get_best_match(user_input)

        if similarity_score >= SIMILARITY_THRESHOLD and best_answer:
            return best_answer, similarity_score

        return FALLBACK_MESSAGE, similarity_score
