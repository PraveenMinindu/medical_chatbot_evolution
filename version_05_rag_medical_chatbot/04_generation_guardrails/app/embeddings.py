"""Embedding model loading and text-to-vector conversion.

This is the core upgrade from A+++.

In A+++ (TF-IDF):
    - Vectors are sparse (most values are zero).
    - Each dimension represents one word from the vocabulary.
    - "hot" and "warm" produce completely different vectors because
      they are different words, even though they mean similar things.

In A++++ (sentence-transformers):
    - Vectors are dense (all 384 values are non-zero).
    - Each dimension represents a learned feature of meaning.
    - "hot" and "warm" produce similar vectors because the model
      learned their semantic relationship from billions of sentences.
    - "I feel feverish" and "my body temperature is high" will have
      similar vectors even though they share no words.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


def load_model() -> SentenceTransformer:
    """Load the sentence-transformer embedding model.

    The model is downloaded once and cached locally by the library.
    Subsequent runs load from cache -- no internet required after first run.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Model loaded.")
    return model


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Convert a list of strings into a matrix of dense embedding vectors.

    Each row in the output matrix is a 384-dimensional vector
    representing the semantic meaning of the corresponding text.

    Args:
        model:  The loaded SentenceTransformer model.
        texts:  List of strings to embed.

    Returns:
        numpy array of shape (len(texts), 384).
    """
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Embed a single query string into a dense vector.

    Args:
        model:  The loaded SentenceTransformer model.
        query:  The user's input string.

    Returns:
        numpy array of shape (384,).
    """
    return model.encode(query, convert_to_numpy=True, show_progress_bar=False)
