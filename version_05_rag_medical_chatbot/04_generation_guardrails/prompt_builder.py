"""Prompt builder: assembles retrieved chunks into a grounded LLM prompt.

WHY PROMPT BUILDING IS THE HEART OF RAG:
    Retrieval finds the right information.
    The prompt tells the LLM what to do with it.

    Without a well-designed prompt, even perfect retrieval produces poor results.
    The LLM might ignore the context and answer from its training data (hallucination).
    Or it might copy chunks verbatim instead of synthesising a useful answer.

HOW GROUNDING WORKS:
    The system prompt contains two critical instructions:
      1. "Answer using ONLY the information in the provided context."
      2. "If the context does not contain enough information, say so clearly."

    These two instructions together create grounding -- the LLM is told both
    what source to use AND what to do when that source is insufficient.

    Grounding does not eliminate hallucination completely, but it dramatically
    reduces it. The LLM is much less likely to invent information when it has
    been given relevant context and explicitly told to use it.

THE STRUCTURE OF A RAG PROMPT:
    A well-designed RAG prompt has three parts:

    [SYSTEM]
        Role definition: who the LLM is and what it can/cannot do.
        Safety constraints: what topics are off-limits.
        Instructions: how to use the context, when to say "I don't know".

    [CONTEXT]
        The retrieved chunks, numbered and labelled with their topic.
        Each chunk is clearly delimited so the LLM can reference them.

    [USER]
        The user's original question, unchanged.

    This structure is called a "grounded prompt" and is standard in RAG systems.
"""

from __future__ import annotations

from app.schemas import RetrievedChunk


# ── System prompt ─────────────────────────────────────────────────────────────
# This is the same for every query. It defines the LLM's role and rules.
# It is kept as a constant string for easy inspection and modification.
SYSTEM_PROMPT = """You are an educational medical information assistant.

Your role:
- Provide clear, accurate, general educational information about common health topics.
- Help users understand symptoms and general health information for learning purposes only.

Your strict rules:
- Answer ONLY using the information provided in the context sections below.
- Do NOT use your own training knowledge to add information not present in the context.
- Do NOT diagnose any condition.
- Do NOT recommend, prescribe, or mention any specific medicine, drug, or dosage.
- Do NOT provide emergency medical instructions.
- If the context does not contain enough information to answer the question, say clearly:
  "I don't have enough information in my knowledge base to answer that question."
- Keep answers concise, clear, and educational.
- Always end with: "This is general educational information only. Please consult a healthcare professional for personal medical advice."

You are not a doctor. You cannot replace professional medical judgement."""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is labelled with its number, topic, and similarity score.
    The topic label helps the LLM understand which area of knowledge each
    chunk belongs to, which can help it synthesise multi-topic answers.

    Args:
        chunks: List of retrieved chunks above the similarity threshold.

    Returns:
        A formatted string containing all chunks as numbered context entries.

    Example output:
        [Context 1] Topic: fever | Score: 0.82
        Fever is a temporary rise in body temperature...

        [Context 2] Topic: dehydration | Score: 0.71
        Dehydration occurs when the body loses more fluid...
    """
    if not chunks:
        return "No relevant context was found in the knowledge base."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[Context {i}] Topic: {chunk.topic} | Score: {chunk.similarity_score:.2f}"
        )
        lines.append(chunk.text)
        lines.append("")   # blank line between chunks

    return "\n".join(lines).strip()


def build_prompt(
    user_question: str,
    chunks: list[RetrievedChunk],
) -> tuple[str, str]:
    """Build a complete grounded RAG prompt from question and retrieved chunks.

    Returns a (system_prompt, user_message) tuple.

    The system prompt is sent as the "system" role in the API call.
    The user message combines the context and the question in the "user" role.

    Why separate system and user:
        OpenAI and Ollama both support multi-role messages (system/user/assistant).
        Putting role instructions in the system role and context+question in the
        user role is the standard pattern. The LLM weights system instructions
        more heavily than user content, making this more reliable than combining
        everything into a single string.

    Args:
        user_question: The raw user query.
        chunks:        Retrieved chunks to use as grounding context.

    Returns:
        Tuple of (system_prompt, user_message).
    """
    context_block = build_context_block(chunks)

    user_message = (
        f"CONTEXT FROM KNOWLEDGE BASE:\n"
        f"{'-' * 50}\n"
        f"{context_block}\n"
        f"{'-' * 50}\n\n"
        f"QUESTION: {user_question}\n\n"
        f"Please answer the question using only the context provided above."
    )

    return SYSTEM_PROMPT, user_message
