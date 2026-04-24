"""Ingestion pipeline -- Day 4 version (imports from app.*)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH, METADATA_PATH, RAW_KNOWLEDGE_PATH,
)
from app.rag.chunking import Chunk, build_chunks
from app.rag.embeddings import embed_texts, load_model
from app.rag.vector_store import build_index


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest medical knowledge into FAISS.")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    p.add_argument("--overlap",    type=int, default=CHUNK_OVERLAP)
    return p.parse_args()


def chunks_to_metadata(chunks: list[Chunk]) -> list[dict]:
    return [
        {"chunk_id": c.chunk_id, "text": c.text, "topic": c.topic,
         "source_file": c.source_file, "char_start": c.char_start,
         "char_end": c.char_end, "chunk_index": c.chunk_index}
        for c in chunks
    ]


def run_ingestion(chunk_size: int, overlap: int) -> None:
    print("=" * 60)
    print("  Day 4 -- Ingestion Pipeline")
    print("=" * 60)

    if not RAW_KNOWLEDGE_PATH.exists():
        print(f"ERROR: {RAW_KNOWLEDGE_PATH} not found.")
        sys.exit(1)

    print(f"\nSource: {RAW_KNOWLEDGE_PATH.name}")
    print(f"Chunk size: {chunk_size}c  |  Overlap: {overlap}c\n")

    chunks = build_chunks(RAW_KNOWLEDGE_PATH, chunk_size, overlap)
    print(f"Chunks produced: {len(chunks)}")
    for topic, count in sorted(Counter(c.topic for c in chunks).items()):
        print(f"  {topic:<35} {count} chunks")

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = chunks_to_metadata(chunks)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nChunks saved: {CHUNKS_PATH}")

    model      = load_model()
    embeddings = embed_texts(model, [c.text for c in chunks])
    print(f"Embeddings: {embeddings.shape}")

    build_index(embeddings, metadata)

    print(f"\nDone. {len(chunks)} chunks indexed.")
    print("Run: python run.py")


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(args.chunk_size, args.overlap)
