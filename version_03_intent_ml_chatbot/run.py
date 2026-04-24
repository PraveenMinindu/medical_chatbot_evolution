from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot import MedicalChatbot
from src.config import DISCLAIMER_MESSAGE, EXIT_COMMANDS, MODEL_PATH, VECTORIZER_PATH
from src.train import train_intent_model


def ensure_model_exists() -> None:
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        return

    print("Model files not found. Training a new intent model...")
    train_intent_model()
    print()


def main() -> None:
    ensure_model_exists()
    chatbot = MedicalChatbot()

    print("Intent-Based Medical Information Chatbot (A++)")
    print(DISCLAIMER_MESSAGE)
    print("Type 'quit', 'exit', or 'bye' to end the chat.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            print("Bot: Please enter a message.")
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print("Bot: Goodbye! Stay healthy.")
            break

        result = chatbot.generate_reply(user_input)
        print(f"Bot: {result['reply']}")
        print(
            f"(Predicted intent: {result['predicted_intent']}, "
            f"confidence: {result['confidence']:.2f})"
        )


if __name__ == "__main__":
    main()
