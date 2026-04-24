"""Shared schemas for internal pipeline and FastAPI layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

ResponseType = Literal["safety", "urgent", "retrieval", "fallback", "error"]


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    topic: str
    source_file: str
    similarity_score: float
    above_threshold: bool


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    has_results: bool
    best_score: float | None


@dataclass
class GenerationResult:
    answer: str
    provider: str
    model: str
    used_fallback: bool


@dataclass
class RAGResponse:
    answer: str
    response_type: ResponseType
    safety_triggered: bool = False
    best_score: float | None = None
    retrieval_scores: list[float] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    topics_found: list[str] = field(default_factory=list)
    generation: GenerationResult | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class GenerationMetaModel(BaseModel):
    provider: str
    model: str
    used_fallback: bool


class RAGResponseModel(BaseModel):
    answer: str
    response_type: ResponseType
    safety_triggered: bool = False
    best_score: float | None = None
    retrieval_scores: list[float] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    topics_found: list[str] = Field(default_factory=list)
    generation: GenerationMetaModel | None = None

    @classmethod
    def from_pipeline(cls, resp: RAGResponse) -> "RAGResponseModel":
        generation = None
        if resp.generation is not None:
            generation = GenerationMetaModel(
                provider=resp.generation.provider,
                model=resp.generation.model,
                used_fallback=resp.generation.used_fallback,
            )
        return cls(
            answer=resp.answer,
            response_type=resp.response_type,
            safety_triggered=resp.safety_triggered,
            best_score=resp.best_score,
            retrieval_scores=resp.retrieval_scores,
            sources=resp.sources,
            topics_found=resp.topics_found,
            generation=generation,
        )

