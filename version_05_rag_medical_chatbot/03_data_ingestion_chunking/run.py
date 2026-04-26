"""CLI entry point for A++++ Day 3 -- chunk-based RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot import MedicalChatbot
from config import EXIT_COMMANDS, WELCOME_MESSAGE
from vector_store import index_exists


def print_response(response) -> None:
    print(f"\nBot: {response.reply}")

    if response.response_type == "retrieval":
        print(f"\n  Best chunk: [{response.best_topic}]  "
              f"score={response.best_score:.3f}  "
              f"source={response.best_source}")

        if len(response.all_chunks) > 1:
            topics_found = list(dict.fromkeys(c.topic for c in response.all_chunks))
            print(f"\n  All retrieved chunks ({len(response.all_chunks)} "
                  f"across {len(topics_found)} topic(s)):")
            for i, chunk in enumerate(response.all_chunks, 1):
                preview = chunk.text[:75].replace("\n", " ")
                print(f"    [{i}] [{chunk.topic:<30}] "
                      f"score={chunk.similarity_score:.3f}")
                print(f"        {preview}...")

    elif response.response_type == "fallback" and response.best_score is not None:
        print(f"  (best score was {response.best_score:.3f} -- below threshold)")

    elif response.response_type == "safety":
        print("  (safety block)")
    print()


def main() -> None:
    print("=" * 62)
    print(WELCOME_MESSAGE)
    print("=" * 62 + "\n")

    if not index_exists():
        print("ERROR: Vector store not found.")
        print("Run this first: python scripts/ingest_data.py\n")
        sys.exit(1)

    chatbot = MedicalChatbot()

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
