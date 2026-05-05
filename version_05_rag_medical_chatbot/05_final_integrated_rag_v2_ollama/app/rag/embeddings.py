"""Embedding model loading and text encoding."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME


def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    return model.encode(query, convert_to_numpy=True, show_progress_bar=False)

