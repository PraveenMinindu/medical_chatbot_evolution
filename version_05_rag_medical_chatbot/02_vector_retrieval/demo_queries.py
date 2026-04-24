"""Day 2 demo: shows top-k multi-topic retrieval in action.

The key demonstration is that a query about multiple symptoms
retrieves results from multiple topics -- exactly what a RAG
system needs before passing context to an LLM.
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
    # Multi-topic -- should retrieve from more than one topic
    ("Multi-topic", "why do I feel hot and weak"),
    ("Multi-topic", "I feel dizzy and my body is burning up"),
    ("Multi-topic", "I have a cold and my head hurts"),

    # Single topic -- should retrieve from one topic cleanly
    ("Single topic", "what is dehydration"),
    ("Single topic", "I cannot stop coughing"),
    ("Single topic", "my head is pounding"),

    # Semantic -- no shared words with KB
    ("Semantic (no word overlap)", "I feel feverish"),
    ("Semantic (no word overlap)", "parched and lightheaded"),

    # Safety -- blocked before retrieval
    ("Safety block", "what medicine should I take for my cough"),
    ("Safety block", "can you diagnose my headache"),

    # Out of scope -- fallback
    ("Out of scope", "what is the best laptop to buy"),
]


def run_demo(chatbot: MedicalChatbot) -> None:
    print("=" * 68)
    print("  DAY 2 DEMO  --  FAISS Top-K Retrieval")
    print("=" * 68)

    for category, query in DEMO_QUERIES:
        print(f"\n[{category}]")
        print(f"  Query: {query!r}")

        response = chatbot.get_response(query)

        print(f"  Type : {response.response_type}")

        if response.response_type == "retrieval":
            print(f"  Best : [{response.best_topic}] "
                  f"score={response.best_score:.3f} "
                  f"({response.best_source_id})")
            print(f"  Reply: {response.reply[:85]}{'...' if len(response.reply) > 85 else ''}")

            if len(response.sources) > 1:
                print(f"  All sources retrieved ({len(response.sources)} topics):")
                for chunk in response.sources:
                    print(f"    -> [{chunk.topic:<15}] "
                          f"score={chunk.similarity_score:.3f}  "
                          f"{chunk.source_id}")

        elif response.response_type == "fallback":
            score_str = f"  Best score: {response.best_score:.3f}" if response.best_score else ""
            print(f"  Reply: {response.reply[:85]}{score_str}")

        elif response.response_type == "safety":
            print(f"  Reply: {response.reply[:85]}")

        print("-" * 68)


if __name__ == "__main__":
    chatbot = MedicalChatbot()
    run_demo(chatbot)
