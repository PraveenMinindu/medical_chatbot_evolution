# run.py
# Medical Chatbot - Version A
# Entry point — runs the conversation loop

from chatbot import get_response


def main():
    print("=" * 55)
    print("       MedBot — Medical Information Chatbot")
    print("              Version A  |  Rule-Based")
    print("=" * 55)
    print("Ask me about: fever, headache, cold, cough, dehydration")
    print("Type 'quit' to exit.")
    print("-" * 55)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Skip if user pressed Enter with nothing
        if not user_input:
            continue

        # Get bot response
        response = get_response(user_input)

        # Check for exit signal
        if response == "EXIT":
            print("\nBot: Goodbye! Stay healthy. 🌿")
            print("-" * 55)
            break

        # Print response
        print(f"\nBot: {response}")
        print("-" * 55)


if __name__ == "__main__":
    main()
