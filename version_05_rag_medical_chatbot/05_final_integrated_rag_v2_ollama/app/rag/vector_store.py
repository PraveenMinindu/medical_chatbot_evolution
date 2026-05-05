"""FAISS vector index build/load/search helpers."""

from __future__ import annotations

import json

import faiss
import numpy as np

from app.config import EMBEDDING_DIMENSION, FAISS_INDEX_PATH, METADATA_PATH, VECTOR_STORE_DIR


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def build_index(embeddings: np.ndarray, metadata: list[dict]) -> faiss.IndexFlatIP:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    normed = _normalise(embeddings.copy())
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(normed)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Vector store not found. Run: python scripts/ingest_data.py")
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return index, metadata


def index_exists() -> bool:
    return FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()


def search(index: faiss.IndexFlatIP, metadata: list[dict], query_vector: np.ndarray, top_k: int) -> list[dict]:
    query = query_vector.astype(np.float32).reshape(1, -1)
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)
    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = dict(metadata[idx])
        entry["similarity_score"] = float(score)
        results.append(entry)
    return results

