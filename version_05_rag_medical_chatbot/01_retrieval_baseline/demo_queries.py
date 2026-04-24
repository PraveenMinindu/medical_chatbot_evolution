"""Demonstration of semantic retrieval vs TF-IDF retrieval.

This script runs a set of test queries through the chatbot and prints
the similarity scores and matched answers. It shows the key advantage
of semantic embeddings: queries that share NO words with KB entries
can still retrieve the correct answer.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.chatbot import MedicalChatbot


DEMO_QUERIES = [
    # These would have scored 0.0 in A+++ TF-IDF because they share no words
    # with the knowledge base questions.
    ("Semantic -- no shared words", "my skull feels like it is burning up"),
    ("Semantic -- synonym",         "I feel feverish"),
    ("Semantic -- synonym",         "I am parched and lightheaded"),
    ("Semantic -- rephrasing",      "my throat keeps making me cough"),
    ("Semantic -- rephrasing",      "fluid loss symptoms"),

    # These would have scored well in A+++ because they share words.
    ("Direct match",  "What is fever?"),
    ("Direct match",  "I have been coughing for days"),
    ("Direct match",  "I feel dizzy and my mouth is dry"),

    # Safety check -- should be blocked before retrieval.
    ("Safety block",  "what medicine should I take for my fever"),
    ("Safety block",  "can you diagnose my headache"),

    # Out of scope -- should fall back.
    ("Out of scope",  "what is the best smartphone to buy"),
]


def run_demo(chatbot: MedicalChatbot) -> None:
    print("=" * 65)
    print("  SEMANTIC RETRIEVAL DEMO  --  A++++ vs A+++")
    print("=" * 65)

    for category, query in DEMO_QUERIES:
        print(f"\n[{category}]")
        print(f"  Query : {query!r}")

        response = chatbot.get_response(query)

        print(f"  Type  : {response.response_type}")
        if response.similarity_score is not None:
            print(f"  Score : {response.similarity_score:.3f}")
        if response.topic:
            print(f"  Topic : {response.topic}")
        print(f"  Reply : {response.reply[:90]}{'...' if len(response.reply) > 90 else ''}")
        print("-" * 65)


if __name__ == "__main__":
    chatbot = MedicalChatbot()
    run_demo(chatbot)
