"""Prompt builder for grounded RAG generation."""

from __future__ import annotations

from app.schemas import RetrievedChunk

SYSTEM_PROMPT = """You are an educational medical information assistant.

Rules:
- Answer ONLY using the provided context.
- Do NOT diagnose.
- Do NOT prescribe medicines, drugs, or dosage.
- Do NOT provide emergency instructions.
- If context is insufficient, say you do not have enough information.
- Keep the answer concise and clear.
- End with: "This is general educational information only. Please consult a healthcare professional for personal medical advice."
"""


def build_context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant context found."
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[Context {i}] Topic: {chunk.topic} | Score: {chunk.similarity_score:.2f}")
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


def build_prompt(user_question: str, chunks: list[RetrievedChunk]) -> tuple[str, str]:
    context = build_context_block(chunks)
    user_message = (
        "CONTEXT FROM KNOWLEDGE BASE:\n"
        + "-" * 50
        + "\n"
        + context
        + "\n"
        + "-" * 50
        + f"\n\nQUESTION: {user_question}\n\n"
        + "Answer using only the context above."
    )
    return SYSTEM_PROMPT, user_message

