"""Top-k chunk retrieval from FAISS index.

Day 3 change from Day 2:
    Day 2 retrieved Q/A pairs and deduplicated by topic.
    Day 3 retrieves text chunks from a real document.

    Each chunk is a passage of ~400 characters from medical_knowledge.md.
    Multiple chunks from the same topic section may all be relevant --
    for example, three different paragraphs about dehydration might each
    contain different but useful information.

    Deduplication logic is updated:
    Day 2: keep only the best-scoring chunk per topic.
    Day 3: keep the best-scoring chunk per topic for display, but pass
           ALL above-threshold chunks as context for the future LLM.
           The LLM needs the full context, not just the single best match.
"""
from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field


from config import (
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from embeddings import embed_query, load_model
from vector_store import index_exists, load_index, search


@dataclass
class RetrievedChunk:
    """One retrieved text chunk with full provenance metadata."""
    chunk_id:         str
    text:             str
    topic:            str
    source_file:      str
    similarity_score: float
    above_threshold:  bool


@dataclass
class RetrievalResponse:
    """Complete result of a top-k chunk retrieval operation.

    best_text:   The text of the highest-scoring relevant chunk.
                 Shown to the user directly (no LLM generation yet).
    best_topic:  Topic of the best chunk.
    best_score:  Similarity score of the best chunk.

    all_chunks:  ALL retrieved chunks above the threshold, sorted by score.
                 This is what Day 4 will pass to the LLM as context.
                 A query about "I feel hot and dizzy" might retrieve:
                   - 2 chunks from fever section
                   - 2 chunks from dehydration section
                 The LLM needs all four to give a complete answer.

    has_results: True if at least one chunk is above the threshold.
    """
    best_text:   str | None
    best_topic:  str | None
    best_source: str | None
    best_score:  float | None
    all_chunks:  list[RetrievedChunk] = field(default_factory=list)
    has_results: bool = False


class KnowledgeRetriever:
    """Loads the FAISS index and retrieves top-k chunks for a query."""

    def __init__(self) -> None:
        if not index_exists():
            raise FileNotFoundError(
                "Vector store not found.\n"
                "Run: python scripts/ingest_data.py"
            )
        self.model = load_model()
        self.index, self.metadata = load_index()

    def retrieve(self, user_query: str, top_k: int = TOP_K) -> RetrievalResponse:
        """Retrieve top-k relevant chunks for a user query.

        Steps:
            1. Embed the user query.
            2. Search FAISS for top_k nearest chunk vectors.
            3. Filter results by similarity threshold.
            4. Return structured RetrievalResponse.

        The all_chunks list preserves ALL relevant chunks across topics,
        because a query may touch multiple topics simultaneously.

        Args:
            user_query: Raw user input string.
            top_k:      Number of chunks to retrieve.

        Returns:
            RetrievalResponse with best chunk and full context list.
        """
        # Step 1: embed query
        query_vector = embed_query(self.model, user_query)

        # Step 2: FAISS search
        raw_results = search(self.index, self.metadata, query_vector, top_k)

        # Step 3: wrap in dataclass and mark threshold status
        chunks = [
            RetrievedChunk(
                chunk_id         = r["chunk_id"],
                text             = r["text"],
                topic            = r["topic"],
                source_file      = r.get("source_file", ""),
                similarity_score = r["similarity_score"],
                above_threshold  = r["similarity_score"] >= SIMILARITY_THRESHOLD,
            )
            for r in raw_results
        ]

        # Step 4: separate valid chunks
        valid_chunks = [c for c in chunks if c.above_threshold]

        if not valid_chunks:
            return RetrievalResponse(
                best_text   = None,
                best_topic  = None,
                best_source = None,
                best_score  = chunks[0].similarity_score if chunks else None,
                all_chunks  = [],
                has_results = False,
            )

        best = valid_chunks[0]

        return RetrievalResponse(
            best_text   = best.text,
            best_topic  = best.topic,
            best_source = best.chunk_id,
            best_score  = best.similarity_score,
            all_chunks  = valid_chunks,   # ALL valid chunks for LLM context
            has_results = True,
        )
