"""CLI entry point for A++++ Day 4 -- Full RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import EXIT_COMMANDS, LLM_PROVIDER, WELCOME_MESSAGE
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import index_exists


def print_response(response) -> None:
    """Print the RAG response with full metadata."""
    print(f"\nBot: {response.answer}")

    if response.response_type == "retrieval" and response.generation:
        g = response.generation
        print(f"\n  Provider:    {g.provider} / {g.model}"
              + (" [fallback used]" if g.used_fallback else ""))
        print(f"  Topics:      {', '.join(response.topics_found)}")
        print(f"  Scores:      {[round(s, 3) for s in response.retrieval_scores]}")
        print(f"  Sources:     {response.sources}")
        if response.safety_triggered:
            print("  [Output safety scan triggered]")

    elif response.response_type == "fallback":
        score = response.retrieval_scores[0] if response.retrieval_scores else None
        if score is not None:
            print(f"  (best score: {score:.3f} -- below threshold)")

    elif response.response_type in ("safety", "urgent"):
        print(f"  [{response.response_type} guardrail triggered]")

    print()


def main() -> None:
    print("=" * 65)
    print(WELCOME_MESSAGE)
    print(f"  LLM Provider: {LLM_PROVIDER}")
    print("=" * 65 + "\n")

    if not index_exists():
        print("ERROR: Vector store not found.")
        print("Run first: python scripts/ingest_data.py\n")
        sys.exit(1)

    pipeline = RAGPipeline()
    print("Pipeline ready.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Bot: Please type a question.\n")
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Bot: Goodbye. Stay informed and take care.")
            break

        response = pipeline.run(user_input)
        print_response(response)


if __name__ == "__main__":
    main()
