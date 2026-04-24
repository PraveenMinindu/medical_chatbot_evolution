"""FAISS vector index: build, persist, load, and search.

Day 3 change from Day 2:
    Day 2 stored Q/A entries with 'question' and 'answer' fields.
    Day 3 stores text chunks with 'text', 'topic', 'chunk_id', etc.
    The FAISS operations are identical -- only the metadata structure changed.

The search() function now returns 'text' instead of 'answer',
and 'chunk_id' instead of 'source_id'.
"""

from __future__ import annotations

import json

import faiss
import numpy as np

from .config import (
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    VECTOR_STORE_DIR,
)


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise so inner product equals cosine similarity."""
    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def build_index(embeddings: np.ndarray, metadata: list[dict]) -> faiss.IndexFlatIP:
    """Build a FAISS IndexFlatIP, save index and metadata to disk."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    normed = _normalise(embeddings.copy())
    index  = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(normed)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"FAISS index saved:     {FAISS_INDEX_PATH}")
    print(f"Metadata saved:        {METADATA_PATH}")
    print(f"Total vectors indexed: {index.ntotal}")
    return index


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load FAISS index and metadata from disk."""
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run: python scripts/ingest_data.py"
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"FAISS index loaded: {index.ntotal} vectors.")
    return index, metadata


def index_exists() -> bool:
    """Return True if saved FAISS index exists."""
    return FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()


def search(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    query_vector: np.ndarray,
    top_k: int,
) -> list[dict]:
    """Search the FAISS index for top_k most similar chunks.

    Returns list of metadata dicts with similarity_score added.
    Each dict contains: chunk_id, text, topic, source_file,
                        char_start, char_end, chunk_index, similarity_score.
    """
    query = query_vector.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query)

    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = dict(metadata[idx])
        entry["similarity_score"] = float(score)
        results.append(entry)

    return results
