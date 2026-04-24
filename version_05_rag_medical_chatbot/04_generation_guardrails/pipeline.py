"""RAG pipeline orchestrator -- the complete flow from query to answer.

This module is what makes the system truly RAG.

WHAT MAKES A SYSTEM TRULY RAG:
    RAG stands for Retrieval-Augmented Generation.
    The three words describe three distinct steps:

    RETRIEVAL:   Find relevant information from an external knowledge source.
                 In this system: FAISS search over embedded document chunks.

    AUGMENTED:   Add that retrieved information to the LLM's input.
                 In this system: the prompt builder inserts chunks into the prompt.

    GENERATION:  The LLM reads the augmented prompt and generates a response.
                 In this system: OpenAI, Ollama, or mock provider.

    Without retrieval: the LLM answers from training data only (prone to hallucination).
    Without generation: the system returns raw chunks verbatim (Days 1-3).
    With both: the LLM synthesises retrieved knowledge into a coherent answer.

THE COMPLETE PIPELINE:
    1. Safety guardrail (input)     -- block before any computation
    2. Retrieval                    -- FAISS top-k semantic search
    3. Fallback check               -- if no results, return fallback message
    4. Prompt building              -- assemble system prompt + context + question
    5. LLM generation               -- call configured LLM provider
    6. Safety guardrail (output)    -- scan generated text for policy violations
    7. Return RAGResponse           -- structured result with full metadata

HOW HALLUCINATION IS REDUCED:
    Step 2 provides the LLM with relevant, factual text from the knowledge base.
    Step 4 instructs the LLM to use ONLY that provided context.
    Step 6 catches any policy-violating content that slipped through.

    The combination of retrieval + grounded prompt + output scanning
    dramatically reduces hallucination compared to a plain LLM call.
    It does not eliminate it -- which is why the system still says
    "please consult a healthcare professional" at the end of every answer.
"""

from __future__ import annotations

from app.config import FALLBACK_MESSAGE, INSUFFICIENT_CONTEXT_MESSAGE
from app.llm.generator import generate
from app.rag.prompt_builder import build_prompt
from app.rag.retrieval import KnowledgeRetriever
from app.safety.guardrails import check_input, scan_output
from app.schemas import GenerationResult, RAGResponse


class RAGPipeline:
    """Orchestrates the full RAG pipeline: safety -> retrieve -> prompt -> generate.

    This class is the single entry point for processing a user query.
    All other modules are called from here in the correct order.

    Usage:
        pipeline = RAGPipeline()
        response = pipeline.run("why do I feel hot and dizzy?")
        print(response.answer)
    """

    def __init__(self) -> None:
        self.retriever = KnowledgeRetriever()

    def run(self, user_query: str) -> RAGResponse:
        """Process a user query through the complete RAG pipeline.

        Args:
            user_query: Raw text from the user.

        Returns:
            RAGResponse with answer, sources, scores, and metadata.
        """
        # ── Step 1: Input safety guardrail ─────────────────────────────────
        # This runs before any embedding or FAISS search.
        # Blocked queries never touch the LLM.
        input_check = check_input(user_query)
        if input_check.blocked:
            return RAGResponse(
                answer           = input_check.safe_response,
                response_type    = "urgent" if input_check.reason == "urgent_phrase" else "safety",
                safety_triggered = True,
            )

        # ── Step 2: Retrieval ───────────────────────────────────────────────
        # Embed the query and find the top-k most similar chunks.
        retrieval = self.retriever.retrieve(user_query)

        # ── Step 3: Fallback if no results above threshold ──────────────────
        if not retrieval.has_results:
            return RAGResponse(
                answer        = FALLBACK_MESSAGE,
                response_type = "fallback",
                best_score    = retrieval.best_score,
            )

        # ── Step 4: Build grounded prompt ───────────────────────────────────
        # Assemble system instructions + numbered context blocks + user question.
        # This is the "augmented" step that grounds the LLM in the retrieved facts.
        system_prompt, user_message = build_prompt(
            user_question = user_query,
            chunks        = retrieval.chunks,
        )

        # ── Step 5: LLM generation ──────────────────────────────────────────
        # Send the grounded prompt to the configured LLM provider.
        generation: GenerationResult = generate(system_prompt, user_message)

        # ── Step 6: Output safety scan ──────────────────────────────────────
        # Check the generated answer for any policy-violating phrases.
        output_check = scan_output(generation.answer)
        if output_check.blocked:
            # Replace unsafe output with safety message
            final_answer     = output_check.safe_response
            safety_triggered = True
        else:
            final_answer     = generation.answer
            safety_triggered = False

        # ── Step 7: Assemble and return the structured response ─────────────
        return RAGResponse(
            answer           = final_answer,
            response_type    = "retrieval",
            safety_triggered = safety_triggered,
            retrieval_scores = [c.similarity_score for c in retrieval.chunks],
            sources          = [c.chunk_id for c in retrieval.chunks],
            topics_found     = list(dict.fromkeys(c.topic for c in retrieval.chunks)),
            generation       = generation,
        )
