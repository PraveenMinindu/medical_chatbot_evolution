"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from app.config import FALLBACK_MESSAGE
from app.llm.generator import generate
from app.rag.prompt_builder import build_prompt
from app.rag.retrieval import KnowledgeRetriever
from app.safety.guardrails import check_input, scan_output
from app.schemas import RAGResponse


class RAGPipeline:
    def __init__(self) -> None:
        self.retriever = KnowledgeRetriever()

    def run(self, user_query: str) -> RAGResponse:
        input_check = check_input(user_query)
        if input_check.blocked:
            return RAGResponse(
                answer=input_check.safe_response or FALLBACK_MESSAGE,
                response_type="urgent" if input_check.reason == "urgent_phrase" else "safety",
                safety_triggered=True,
            )

        retrieval = self.retriever.retrieve(user_query)
        if not retrieval.has_results:
            return RAGResponse(
                answer=FALLBACK_MESSAGE,
                response_type="fallback",
                best_score=retrieval.best_score,
            )

        system_prompt, user_message = build_prompt(user_query, retrieval.chunks)
        generation = generate(system_prompt, user_message)

        output_check = scan_output(generation.answer)
        if output_check.blocked:
            final_answer = output_check.safe_response or FALLBACK_MESSAGE
            safety_triggered = True
        else:
            final_answer = generation.answer
            safety_triggered = False

        return RAGResponse(
            answer=final_answer,
            response_type="retrieval",
            safety_triggered=safety_triggered,
            best_score=retrieval.best_score,
            retrieval_scores=[c.similarity_score for c in retrieval.chunks],
            sources=[c.chunk_id for c in retrieval.chunks],
            topics_found=list(dict.fromkeys(c.topic for c in retrieval.chunks)),
            generation=generation,
        )

