# run.py
# Medical Chatbot - Version A+
# Entry point — runs the conversation loop

from chatbot import get_response, get_all_matches
from utils import clean_text
from keywords import unsafe_keywords, exit_words, greetings, unsafe_response
from utils import extract_keywords, is_whole_word


def main():
    print("=" * 58)
    print("       MedBot A+ — Medical Information Chatbot")
    print("          Version A+  |  Keyword-Based NLP")
    print("=" * 58)
    print("Speak naturally! Try: 'I have fever and a headache'")
    print("Type 'quit' to exit.")
    print("-" * 58)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        cleaned = clean_text(user_input)

        # Check exit first
        response = get_response(user_input)
        if response == "EXIT":
            print("\nBot: Goodbye! Stay healthy. 🌿")
            print("-" * 58)
            break

        # Determine if this is a special case (greeting/unsafe)
        is_special = (
            bool(extract_keywords(cleaned, greetings)) or
            bool(extract_keywords(cleaned, unsafe_keywords))
        )

        # For normal medical queries, try multi-topic detection
        if not is_special:
            all_matches = get_all_matches(user_input)
            if len(all_matches) > 1:
                print(f"\nBot: I found {len(all_matches)} topics in your message:\n")
                for i, (topic, resp) in enumerate(all_matches, 1):
                    print(f"  [{i}] {resp}")
                    print()
                print("-" * 58)
                continue

        # Single response
        print(f"\nBot: {response}")
        print("-" * 58)


if __name__ == "__main__":
    main()
