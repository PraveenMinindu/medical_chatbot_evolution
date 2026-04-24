"""Top-k chunk retrieval from FAISS index -- Day 4 version.

Updated for Day 4: imports from app.* instead of src.*
The retrieval logic is unchanged from Day 3.
"""

from __future__ import annotations

from app.config import SIMILARITY_THRESHOLD, TOP_K
from app.rag.embeddings import embed_query, load_model
from app.rag.vector_store import index_exists, load_index, search
from app.schemas import RetrievalResult, RetrievedChunk


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

    def retrieve(self, user_query: str, top_k: int = TOP_K) -> RetrievalResult:
        """Retrieve top-k relevant chunks for a user query.

        Returns a RetrievalResult with all chunks above the threshold.
        All valid chunks are passed as context to the LLM prompt builder.
        """
        query_vector = embed_query(self.model, user_query)
        raw_results  = search(self.index, self.metadata, query_vector, top_k)

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

        valid_chunks = [c for c in chunks if c.above_threshold]

        return RetrievalResult(
            chunks      = valid_chunks,
            has_results = bool(valid_chunks),
            best_score  = valid_chunks[0].similarity_score if valid_chunks else (
                chunks[0].similarity_score if chunks else None
            ),
        )
