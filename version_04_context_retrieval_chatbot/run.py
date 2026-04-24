"""Command-line entry point for the retrieval-based medical chatbot."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot import MedicalChatbot  # noqa: E402
from config import EXIT_COMMANDS  # noqa: E402


def main() -> None:
    chatbot = MedicalChatbot()

    print("Knowledge-Based Medical Information Chatbot (A+++)")
    print("Ask about fever, headache, cough, cold, or dehydration.")
    print("Type quit, exit, or bye to stop.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Bot: Please enter a question.\n")
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Bot: Goodbye. Stay informed and take care.")
            break

        answer, _ = chatbot.get_response(user_input)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()
