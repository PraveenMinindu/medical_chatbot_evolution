"""Ingest raw medical document, chunk, embed, and build FAISS index."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH, RAW_KNOWLEDGE_PATH
from app.rag.chunking import Chunk, build_chunks
from app.rag.embeddings import embed_texts, load_model
from app.rag.vector_store import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest medical knowledge into FAISS")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    return parser.parse_args()


def to_metadata(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "topic": c.topic,
            "source_file": c.source_file,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]


def run_ingestion(chunk_size: int, overlap: int) -> None:
    if not RAW_KNOWLEDGE_PATH.exists():
        raise FileNotFoundError(f"Source file not found: {RAW_KNOWLEDGE_PATH}")

    chunks = build_chunks(RAW_KNOWLEDGE_PATH, chunk_size, overlap)
    metadata = to_metadata(chunks)

    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    model = load_model()
    embeddings = embed_texts(model, [c.text for c in chunks])
    build_index(embeddings, metadata)

    print(f"Chunks created: {len(chunks)}")
    for topic, count in sorted(Counter(c.topic for c in chunks).items()):
        print(f"  {topic}: {count}")
    print("Ingestion complete. Run: python run.py")


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(args.chunk_size, args.overlap)

