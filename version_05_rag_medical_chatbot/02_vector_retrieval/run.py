"""CLI entry point for A++++ Day 2 -- FAISS vector index chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot import MedicalChatbot
from config import EXIT_COMMANDS, WELCOME_MESSAGE


def print_response(response) -> None:
    """Print the chatbot response with retrieval metadata."""
    print(f"\nBot: {response.reply}")

    if response.response_type == "retrieval":
        print(f"\n  Best match:  [{response.best_topic}]  "
              f"score={response.best_score:.3f}  "
              f"source={response.best_source_id}")

        if len(response.sources) > 1:
            print(f"\n  All retrieved sources ({len(response.sources)} topics):")
            for i, chunk in enumerate(response.sources, 1):
                print(f"    [{i}] topic={chunk.topic:<15} "
                      f"score={chunk.similarity_score:.3f}  "
                      f"source={chunk.source_id}")
                print(f"        {chunk.answer[:80]}{'...' if len(chunk.answer) > 80 else ''}")

    elif response.response_type == "fallback" and response.best_score is not None:
        print(f"  (best score was {response.best_score:.3f} -- below threshold)")

    elif response.response_type == "safety":
        print("  (safety block)")

    print()


def main() -> None:
    # --rebuild flag forces the FAISS index to be rebuilt from scratch
    rebuild = "--rebuild" in sys.argv

    print("=" * 62)
    print(WELCOME_MESSAGE)
    if rebuild:
        print("  [--rebuild flag detected: rebuilding FAISS index]")
    print("=" * 62 + "\n")

    chatbot = MedicalChatbot(rebuild=rebuild)

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Bot: Please type a question.\n")
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Bot: Goodbye. Stay informed and take care.")
            break

        response = chatbot.get_response(user_input)
        print_response(response)


if __name__ == "__main__":
    main()
