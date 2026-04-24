"""CLI entry point for the Part 5 final integrated RAG chatbot."""

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
    print(f"\nBot: {response.answer}\n")
    if response.response_type == "retrieval":
        print(f"  Provider: {response.generation.provider}/{response.generation.model}" if response.generation else "  Provider: n/a")
        print(f"  Best score: {response.best_score:.3f}" if response.best_score is not None else "  Best score: n/a")
        print(f"  Topics: {', '.join(response.topics_found) if response.topics_found else 'n/a'}")
        print(f"  Sources: {response.sources}\n")
    elif response.response_type == "fallback" and response.best_score is not None:
        print(f"  Best score below threshold: {response.best_score:.3f}\n")


def main() -> None:
    print("=" * 72)
    print(WELCOME_MESSAGE)
    print(f"LLM Provider: {LLM_PROVIDER}")
    print("=" * 72)

    if not index_exists():
        print("\nVector store missing. Run first: python scripts/ingest_data.py\n")
        sys.exit(1)

    pipeline = RAGPipeline()
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

