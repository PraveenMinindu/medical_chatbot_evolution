"""Command-line entry point for the A++++ semantic medical chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chatbot import MedicalChatbot
from config import EXIT_COMMANDS, WELCOME_MESSAGE


def main() -> None:
    print("=" * 60)
    print(WELCOME_MESSAGE)
    print("=" * 60 + "\n")

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

        print(f"\nBot: {response.reply}")

        # Show retrieval metadata so the user can see how the system works
        if response.similarity_score is not None:
            print(f"     (similarity: {response.similarity_score:.3f}"
                  f"  |  type: {response.response_type}"
                  + (f"  |  topic: {response.topic}" if response.topic else "")
                  + ")")
        else:
            print(f"     (type: {response.response_type})")

        print()


if __name__ == "__main__":
    main()
