"""Retrieval module built with TF-IDF and cosine similarity."""

from __future__ import annotations

from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import KNOWLEDGE_BASE_PATH
from utils import load_knowledge_base, preprocess_text


class KnowledgeRetriever:
    """Vectorizes the knowledge base and retrieves the closest answer."""

    def __init__(self, knowledge_base_path=KNOWLEDGE_BASE_PATH) -> None:
        self.knowledge_base = load_knowledge_base(knowledge_base_path)
        self.vectorizer = self.build_vectorizer()
        self.question_vectors = self.vectorize_knowledge()

    def build_vectorizer(self) -> TfidfVectorizer:
        """Create the TF-IDF vectorizer."""
        return TfidfVectorizer(ngram_range=(1, 2), stop_words="english")

    def vectorize_knowledge(self):
        """Fit the vectorizer on knowledge-base questions."""
        questions = [preprocess_text(item["question"]) for item in self.knowledge_base]
        return self.vectorizer.fit_transform(questions)

    def get_best_match(self, user_input: str) -> tuple[Optional[str], float]:
        """Return the best matching answer and similarity score."""
        cleaned_input = preprocess_text(user_input)
        user_vector = self.vectorizer.transform([cleaned_input])
        similarity_scores = cosine_similarity(user_vector, self.question_vectors).flatten()

        best_index = int(similarity_scores.argmax())
        best_score = float(similarity_scores[best_index])
        best_answer = self.knowledge_base[best_index]["answer"]
        return best_answer, best_score


def build_vectorizer() -> TfidfVectorizer:
    """Module-level helper requested for the assignment structure."""
    return KnowledgeRetriever().build_vectorizer()


def vectorize_knowledge():
    """Module-level helper requested for the assignment structure."""
    return KnowledgeRetriever().vectorize_knowledge()


def get_best_match(user_input: str) -> tuple[Optional[str], float]:
    """Module-level helper requested for the assignment structure."""
    retriever = KnowledgeRetriever()
    return retriever.get_best_match(user_input)
