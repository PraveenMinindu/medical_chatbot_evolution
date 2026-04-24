"""Shared data schemas used across the RAG pipeline.

Having schemas in one place means every module uses the same data shapes.
This becomes essential when Day 5 adds a FastAPI layer -- FastAPI's
request/response validation uses these same structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RetrievedChunk:
    """One text chunk retrieved from the FAISS index."""
    chunk_id:         str
    text:             str
    topic:            str
    source_file:      str
    similarity_score: float
    above_threshold:  bool


@dataclass
class RetrievalResult:
    """Output of the retrieval step -- passed to the prompt builder."""
    chunks:      list[RetrievedChunk]
    has_results: bool
    best_score:  float | None


@dataclass
class GenerationResult:
    """Output of the LLM generation step."""
    answer:        str
    provider:      str    # "openai" | "ollama" | "mock"
    model:         str
    used_fallback: bool   # True if LLM failed and a fallback string was used


ResponseType = Literal["safety", "urgent", "retrieval", "fallback", "error"]


@dataclass
class RAGResponse:
    """The complete response returned by the RAG pipeline for one user query.

    This is the final output object. Day 5 will serialise this directly
    into a FastAPI JSON response.

    Fields:
        answer:          The final answer shown to the user.
        response_type:   How this response was generated.
        safety_triggered: True if a keyword guardrail blocked the query.
        retrieval_scores: Similarity scores of retrieved chunks.
        sources:         chunk_ids of the retrieved chunks used as context.
        topics_found:    Unique topics in the retrieved chunks.
        generation:      Metadata about the LLM call (provider, model, fallback).
    """
    answer:            str
    response_type:     ResponseType
    safety_triggered:  bool                    = False
    retrieval_scores:  list[float]             = field(default_factory=list)
    sources:           list[str]               = field(default_factory=list)
    topics_found:      list[str]               = field(default_factory=list)
    generation:        GenerationResult | None = None
