"""FAISS vector index: build, persist, load, and search.

WHY FAISS INSTEAD OF CHROMA FOR THIS PROJECT:
    Chroma is easier to set up for beginners but adds a server/client layer
    and several dependencies. FAISS is a single C++ library with a Python wrapper.
    It is lower-level, which means more to understand -- and that is exactly what
    a learning project should use. It is also what production systems at companies
    like Meta use at scale.

WHAT A VECTOR INDEX IS:
    In Day 1, we stored embeddings in a numpy array and looped through every
    entry to compute similarity. This is called a brute-force flat search.
    It works fine for 30 entries. For 1 million entries it would take seconds
    per query.

    A vector index organises the vectors into a data structure that allows
    much faster search. FAISS builds this structure once and then finds the
    nearest neighbours in sub-linear time.

    For this project (30 entries) the speed difference is invisible. The value
    is learning the pattern that scales to production.

HOW COSINE SIMILARITY WORKS WITH FAISS:
    FAISS natively computes L2 (Euclidean) distance, not cosine similarity.
    However, if we L2-normalise every vector before storing it, then L2
    distance and cosine similarity produce the same ranking.

    Proof:
        For unit vectors A and B (||A|| = ||B|| = 1):
        ||A - B||^2 = 2 - 2 * dot(A, B) = 2 - 2 * cosine_similarity(A, B)

        Minimising L2 distance is equivalent to maximising cosine similarity
        when all vectors are normalised.

    We use IndexFlatIP (Inner Product) after normalisation, which directly
    computes cosine similarity as a dot product between unit vectors.
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
    """L2-normalise a matrix of vectors so each row has unit length.

    After normalisation, dot product == cosine similarity.
    FAISS IndexFlatIP computes dot products, so this gives us cosine similarity.

    Args:
        vectors: numpy array of shape (n, embedding_dim). Modified in place
                 by faiss.normalize_L2 for efficiency.

    Returns:
        The normalised array (same object, modified in place by FAISS).
    """
    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def build_index(
    embeddings: np.ndarray,
    metadata: list[dict],
) -> faiss.IndexFlatIP:
    """Build a FAISS IndexFlatIP from embedding vectors and save to disk.

    IndexFlatIP = Flat (brute-force) index using Inner Product similarity.
    After normalisation this equals cosine similarity.

    For a student project with ~30 entries, flat search is perfectly adequate.
    For 100,000+ entries, Day 3+ would upgrade to IndexIVFFlat (approximate
    nearest neighbour) which is faster but slightly less accurate.

    Args:
        embeddings: numpy array of shape (n, EMBEDDING_DIMENSION).
        metadata:   list of dicts with source_id, topic, question, answer
                    for each entry -- one dict per embedding row.

    Returns:
        The built FAISS index (also saved to disk).
    """
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Normalise so inner product == cosine similarity
    normed = _normalise(embeddings.copy())

    # Build and populate the index
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(normed)

    # Save the FAISS index binary to disk
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    # Save metadata as JSON -- FAISS stores only vectors, not the text/labels.
    # We keep metadata separately, indexed by the same integer positions.
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"FAISS index saved: {FAISS_INDEX_PATH}")
    print(f"Metadata saved:    {METADATA_PATH}")
    print(f"Total vectors indexed: {index.ntotal}")

    return index


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Load a previously saved FAISS index and its metadata from disk.

    Returns:
        Tuple of (faiss_index, metadata_list).

    Raises:
        FileNotFoundError: if index or metadata files do not exist yet.
    """
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run with --rebuild to create it."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"FAISS index loaded: {index.ntotal} vectors.")
    return index, metadata


def index_exists() -> bool:
    """Return True if a saved FAISS index exists on disk."""
    return FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()


def search(
    index: faiss.IndexFlatIP,
    metadata: list[dict],
    query_vector: np.ndarray,
    top_k: int,
) -> list[dict]:
    """Search the FAISS index for the top_k most similar entries.

    Args:
        index:        The loaded FAISS index.
        metadata:     Metadata list aligned with index positions.
        query_vector: 1D numpy array of shape (EMBEDDING_DIMENSION,).
        top_k:        Number of nearest neighbours to retrieve.

    Returns:
        List of dicts, each containing:
            source_id, topic, question, answer, similarity_score.
        Sorted by similarity_score descending.
    """
    # Normalise query vector to unit length (same as stored vectors)
    query = query_vector.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query)

    # FAISS search returns:
    #   scores: (1, top_k) -- inner product (= cosine similarity after norm)
    #   indices: (1, top_k) -- integer positions in the index
    scores, indices = index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 for padding when fewer than top_k results exist
            continue
        entry = metadata[idx]
        results.append({
            "source_id":        entry["source_id"],
            "topic":            entry["topic"],
            "question":         entry["question"],
            "answer":           entry["answer"],
            "similarity_score": float(score),
        })

    # Results are already sorted by FAISS (highest score first)
    return results
