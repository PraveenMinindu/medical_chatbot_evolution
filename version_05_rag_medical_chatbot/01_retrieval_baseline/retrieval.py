"""Knowledge base retrieval using dense semantic embeddings.

This replaces the TF-IDF + cosine similarity pipeline from A+++.

Key difference:
    A+++ computed similarity over sparse TF-IDF vectors.
          "coughing" and "cough" were different dimensions -- no match.

    A++++ computes similarity over dense semantic vectors.
          "coughing" and "cough" produce similar embeddings.
          "I feel feverish" matches "I have a fever" even with no shared words.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from .config import KNOWLEDGE_BASE_PATH, SIMILARITY_THRESHOLD
from .embeddings import embed_query, embed_texts, load_model
from .utils import load_knowledge_base


@dataclass
class RetrievalResult:
    """Holds the result of a single retrieval operation."""
    answer: str
    question: str
    topic: str
    source_id: str
    similarity_score: float
    above_threshold: bool


class KnowledgeRetriever:
    """Loads the knowledge base, embeds all entries, and retrieves best matches.

    Workflow:
        1. Load all KB entries from JSON.
        2. Embed every question using the sentence-transformer model.
        3. Store the resulting dense vector matrix in memory.
        4. For each user query: embed the query and compute cosine similarity
           against the stored matrix. Return the highest-scoring entry.
    """

    def __init__(self) -> None:
        self.model: SentenceTransformer = load_model()
        self.kb: list[dict] = load_knowledge_base(KNOWLEDGE_BASE_PATH)
        self.questions: list[str] = [entry["question"] for entry in self.kb]

        print(f"Embedding {len(self.questions)} knowledge base entries...")
        # Shape: (num_entries, 384)
        self.kb_embeddings: np.ndarray = embed_texts(self.model, self.questions)
        print("Knowledge base ready.\n")

    def get_best_match(self, user_query: str) -> RetrievalResult:
        """Find the most semantically similar KB entry for the user query.

        Steps:
            1. Embed the user query into a dense vector.
            2. Compute cosine similarity between query vector and all KB vectors.
            3. Find the index of the highest similarity score.
            4. Return the corresponding KB entry with metadata.

        Args:
            user_query: Raw user input string.

        Returns:
            RetrievalResult dataclass with answer, metadata, and score.
        """
        # Step 1: embed query -- shape (384,) reshaped to (1, 384) for sklearn
        query_vector = embed_query(self.model, user_query).reshape(1, -1)

        # Step 2: cosine similarity between query and all KB question vectors
        # similarity_scores shape: (1, num_entries) -- we flatten to (num_entries,)
        similarity_scores = cosine_similarity(query_vector, self.kb_embeddings).flatten()

        # Step 3: find best match
        best_index = int(np.argmax(similarity_scores))
        best_score = float(similarity_scores[best_index])
        best_entry = self.kb[best_index]

        return RetrievalResult(
            answer=best_entry["answer"],
            question=best_entry["question"],
            topic=best_entry["topic"],
            source_id=best_entry["source_id"],
            similarity_score=best_score,
            above_threshold=best_score >= SIMILARITY_THRESHOLD,
        )
