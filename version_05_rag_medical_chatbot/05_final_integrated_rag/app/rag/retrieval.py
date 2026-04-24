"""Top-k retrieval over FAISS with threshold filtering."""

from __future__ import annotations

from app.config import SIMILARITY_THRESHOLD, TOP_K
from app.rag.embeddings import embed_query, load_model
from app.rag.vector_store import index_exists, load_index, search
from app.schemas import RetrievalResult, RetrievedChunk


class KnowledgeRetriever:
    def __init__(self) -> None:
        if not index_exists():
            raise FileNotFoundError("Vector store not found. Run: python scripts/ingest_data.py")
        self.model = load_model()
        self.index, self.metadata = load_index()

    def retrieve(self, user_query: str, top_k: int = TOP_K) -> RetrievalResult:
        query_vector = embed_query(self.model, user_query)
        raw = search(self.index, self.metadata, query_vector, top_k)
        chunks = [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                topic=item["topic"],
                source_file=item.get("source_file", ""),
                similarity_score=item["similarity_score"],
                above_threshold=item["similarity_score"] >= SIMILARITY_THRESHOLD,
            )
            for item in raw
        ]
        valid = [c for c in chunks if c.above_threshold]
        best_score = valid[0].similarity_score if valid else (chunks[0].similarity_score if chunks else None)
        return RetrievalResult(chunks=valid, has_results=bool(valid), best_score=best_score)

