"""Data ingestion pipeline: document -> chunks -> embeddings -> FAISS index.

This script is the entry point for building the knowledge store from raw documents.
It replaces the old approach of manually writing Q/A pairs in a JSON file.

Pipeline:
    1. Load raw document (data/raw/medical_knowledge.md)
    2. Extract sections by heading
    3. Split each section into overlapping text chunks
    4. Save chunks to data/processed/chunks.json (for inspection)
    5. Embed all chunk texts using sentence-transformers
    6. Build a FAISS index from the embeddings
    7. Save index and metadata to vector_store/

Run this script once before using run.py.
Run it again whenever the raw document changes.

Usage:
    python scripts/ingest_data.py
    python scripts/ingest_data.py --chunk-size 500 --overlap 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chunking import Chunk, build_chunks
from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_PATH,
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    RAW_KNOWLEDGE_PATH,
    VECTOR_STORE_DIR,
)
from src.embeddings import embed_texts, load_model
from src.vector_store import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest medical knowledge document into FAISS vector store."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Target chunk size in characters (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlap between chunks in characters (default: {CHUNK_OVERLAP})",
    )
    return parser.parse_args()


def chunks_to_metadata(chunks: list[Chunk]) -> list[dict]:
    """Convert Chunk dataclass objects to plain dicts for JSON and FAISS metadata."""
    return [
        {
            "chunk_id":    c.chunk_id,
            "text":        c.text,
            "topic":       c.topic,
            "source_file": c.source_file,
            "char_start":  c.char_start,
            "char_end":    c.char_end,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]


def save_chunks_json(chunks: list[Chunk], output_path: Path) -> None:
    """Save all chunks to a readable JSON file for inspection and debugging.

    This file is not used at runtime -- it is for the developer to inspect
    what the chunker produced before committing to an index.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = chunks_to_metadata(chunks)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Chunks saved to: {output_path}")


def run_ingestion(chunk_size: int, overlap: int) -> None:
    print("=" * 60)
    print("  Day 3 Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Check source document exists
    if not RAW_KNOWLEDGE_PATH.exists():
        print(f"ERROR: Source document not found: {RAW_KNOWLEDGE_PATH}")
        sys.exit(1)
    print(f"\nSource document: {RAW_KNOWLEDGE_PATH.name}")
    print(f"Chunk size:      {chunk_size} characters")
    print(f"Overlap:         {overlap} characters\n")

    # Step 2: Chunk the document
    print("Chunking document...")
    chunks = build_chunks(RAW_KNOWLEDGE_PATH, chunk_size, overlap)
    print(f"Produced {len(chunks)} chunks across "
          f"{len(set(c.topic for c in chunks))} topics.")

    # Print a summary table of chunks per topic
    from collections import Counter
    topic_counts = Counter(c.topic for c in chunks)
    for topic, count in sorted(topic_counts.items()):
        print(f"  {topic:<30} {count} chunks")

    # Step 3: Save chunks JSON (for inspection)
    save_chunks_json(chunks, CHUNKS_PATH)

    # Step 4: Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL_NAME}")
    model = load_model()

    # Step 5: Embed all chunk texts
    print(f"Embedding {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(model, texts)
    print(f"Embeddings shape: {embeddings.shape}")

    # Step 6: Build FAISS index
    print("\nBuilding FAISS index...")
    metadata = chunks_to_metadata(chunks)
    build_index(embeddings, metadata)

    print("\n" + "=" * 60)
    print(f"  Ingestion complete.")
    print(f"  {len(chunks)} chunks indexed.")
    print(f"  FAISS index: {FAISS_INDEX_PATH}")
    print(f"  Metadata:    {METADATA_PATH}")
    print("=" * 60)
    print("\nYou can now run: python run.py")


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(chunk_size=args.chunk_size, overlap=args.overlap)
