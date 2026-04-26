"""Top-k retrieval from FAISS vector index with deduplication.

Day 1 retrieval:   return the single best match.
Day 2 retrieval:   return the top_k best matches, deduplicated.

WHY TOP-K RETRIEVAL MATTERS:
    A query like "why do I feel hot and weak" is relevant to both fever
    and dehydration. Returning only the best single match loses half the
    relevant information. A RAG system needs multiple retrieved chunks to
    give the LLM enough context to generate a complete, accurate answer.

    top_k retrieval is the standard in all production RAG systems.
    Typical values: k=3 to k=10 depending on context window size.

DEDUPLICATION:
    When multiple KB entries cover the same topic (e.g., "What is fever?"
    and "my body feels hot" both score high for a fever query), returning
    all of them is redundant. We deduplicate by keeping only the highest-
    scoring result per topic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np

from config import (
    DEDUP_SCORE_DELTA,
    KNOWLEDGE_BASE_PATH,
    SIMILARITY_THRESHOLD,
    TOP_K,
    VECTOR_STORE_DIR,
    EMBEDDING_DIMENSION,
    FAISS_INDEX_PATH,
    METADATA_PATH,
)
from embeddings import embed_query, embed_texts, load_model
from utils import load_knowledge_base
from vector_store import build_index, index_exists, load_index, search


@dataclass
class RetrievedChunk:
    """One retrieved knowledge base entry with full metadata."""
    source_id:        str
    topic:            str
    question:         str
    answer:           str
    similarity_score: float
    above_threshold:  bool


@dataclass
class RetrievalResponse:
    """The complete result of a top-k retrieval operation.

    This is what gets passed to the chatbot layer (and eventually to an LLM).
    It contains the best single answer plus all retrieved chunks as context.
    """
    # The single best answer (highest scoring chunk above threshold)
    best_answer:    str | None
    best_source_id: str | None
    best_topic:     str | None
    best_score:     float | None

    # All retrieved chunks above threshold, sorted by score
    # This is what a future LLM would receive as its context window
    top_chunks:     list[RetrievedChunk] = field(default_factory=list)

    # True if at least one result is above the similarity threshold
    has_results:    bool = False


class KnowledgeRetriever:
    """Manages the FAISS index and runs top-k semantic retrieval.

    On first run (or when --rebuild is passed):
        - Loads the KB from JSON
        - Embeds all entries using sentence-transformers
        - Builds a FAISS IndexFlatIP and saves it to vector_store/

    On subsequent runs:
        - Loads the saved FAISS index and metadata from disk
        - Skips re-embedding entirely (fast startup)
    """

    def __init__(self, rebuild: bool = False) -> None:
        self.model = load_model()

        if not rebuild and index_exists():
            # Fast path: load pre-built index from disk
            self.index, self.metadata = load_index()
        else:
            # Build path: embed all KB entries and create fresh index
            print("Building FAISS index from knowledge base...")
            kb = load_knowledge_base(KNOWLEDGE_BASE_PATH)

            questions = [entry["question"] for entry in kb]
            embeddings = embed_texts(self.model, questions)

            # Metadata stored alongside the index (FAISS only stores vectors)
            metadata = [
                {
                    "source_id": e["source_id"],
                    "topic":     e["topic"],
                    "question":  e["question"],
                    "answer":    e["answer"],
                }
                for e in kb
            ]

            self.index    = build_index(embeddings, metadata)
            self.metadata = metadata
            print("Index built and saved.\n")

    def retrieve(self, user_query: str, top_k: int = TOP_K) -> RetrievalResponse:
        """Embed query, search FAISS, deduplicate, and return structured results.

        Args:
            user_query: Raw user input string.
            top_k:      Number of candidates to retrieve from FAISS.

        Returns:
            RetrievalResponse with best answer and all relevant chunks.
        """
        # Step 1: embed the user query
        query_vector = embed_query(self.model, user_query)

        # Step 2: search FAISS for top_k candidates
        raw_results = search(self.index, self.metadata, query_vector, top_k)

        # Step 3: wrap results in RetrievedChunk dataclass
        chunks = [
            RetrievedChunk(
                source_id        = r["source_id"],
                topic            = r["topic"],
                question         = r["question"],
                answer           = r["answer"],
                similarity_score = r["similarity_score"],
                above_threshold  = r["similarity_score"] >= SIMILARITY_THRESHOLD,
            )
            for r in raw_results
        ]

        # Step 4: filter to only chunks above the threshold
        valid_chunks = [c for c in chunks if c.above_threshold]

        # Step 5: deduplicate by topic
        # Keep only the highest-scoring chunk per topic.
        # This prevents the response from repeating information about the same topic.
        valid_chunks = _deduplicate_by_topic(valid_chunks)

        if not valid_chunks:
            return RetrievalResponse(
                best_answer    = None,
                best_source_id = None,
                best_topic     = None,
                best_score     = chunks[0].similarity_score if chunks else None,
                top_chunks     = [],
                has_results    = False,
            )

        # The first chunk is the highest scoring (FAISS returns sorted results)
        best = valid_chunks[0]

        return RetrievalResponse(
            best_answer    = best.answer,
            best_source_id = best.source_id,
            best_topic     = best.topic,
            best_score     = best.similarity_score,
            top_chunks     = valid_chunks,
            has_results    = True,
        )


def _deduplicate_by_topic(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep only the highest-scoring chunk per topic.

    Example:
        Input:  [fever_004 (0.82), fever_001 (0.79), dehydration_001 (0.61)]
        Output: [fever_004 (0.82), dehydration_001 (0.61)]

    Why: If both "my body feels hot" and "What is fever?" score highly for
    the same query, they contain overlapping information. A future LLM only
    needs one good chunk per topic, not duplicates.

    Args:
        chunks: List of RetrievedChunk sorted by similarity_score descending.

    Returns:
        Deduplicated list, preserving score order.
    """
    seen_topics: dict[str, float] = {}
    result: list[RetrievedChunk] = []

    for chunk in chunks:
        prev_score = seen_topics.get(chunk.topic)
        if prev_score is None:
            # First time seeing this topic -- keep it
            seen_topics[chunk.topic] = chunk.similarity_score
            result.append(chunk)
        # If the topic was already seen, we skip this chunk.
        # Since chunks are sorted descending, the first occurrence is the best.

    return result
