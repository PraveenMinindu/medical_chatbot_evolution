"""FastAPI app for the final integrated RAG chatbot."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import index_exists
from app.schemas import QueryRequest, RAGResponseModel

app = FastAPI(
    title="Medical Chatbot Final Integrated RAG",
    version="1.0.0",
    description="Part 5: safety + retrieval + generation API",
)


def _build_pipeline() -> RAGPipeline:
    if not index_exists():
        raise HTTPException(status_code=503, detail="Vector index not found. Run scripts/ingest_data.py first.")
    return RAGPipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": index_exists()}


@app.post("/query", response_model=RAGResponseModel)
def query(payload: QueryRequest) -> RAGResponseModel:
    pipeline = _build_pipeline()
    response = pipeline.run(payload.query)
    return RAGResponseModel.from_pipeline(response)

