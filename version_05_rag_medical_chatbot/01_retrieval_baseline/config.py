"""Configuration for the A++++ semantic embedding chatbot."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.json"

# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2 is small (80 MB), fast, and works well for semantic similarity.
# It produces 384-dimensional dense vectors.
# This is the standard starting model for semantic search projects.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Minimum cosine similarity score required to return a KB answer.
# Range: 0.0 (no similarity) to 1.0 (identical).
# 0.35 is a reasonable starting point for medical Q&A retrieval.
SIMILARITY_THRESHOLD = 0.35

# ── Messages ──────────────────────────────────────────────────────────────────
FALLBACK_MESSAGE = (
    "Sorry, I can only provide general educational information about "
    "fever, headache, cough, cold, and dehydration. "
    "Please rephrase your question or ask about one of those topics."
)

SAFETY_MESSAGE = (
    "I can only provide general educational information. "
    "Please consult a qualified healthcare professional for personal medical advice."
)

WELCOME_MESSAGE = (
    "Semantic Medical Information Chatbot (A++++)\n"
    "Powered by sentence-transformer embeddings.\n"
    "Ask about fever, headache, cough, cold, or dehydration.\n"
    "Type 'quit', 'exit', or 'bye' to stop."
)

# ── Safety ────────────────────────────────────────────────────────────────────
# These keywords trigger an immediate safety response before retrieval runs.
RESTRICTED_KEYWORDS = {
    "diagnose", "diagnosis",
    "prescribe", "prescription",
    "medicine", "medication",
    "drug", "drugs",
    "tablet", "tablets",
    "antibiotic", "antibiotics",
    "dosage", "dose",
    "treatment", "treat",
    "cure", "surgery",
    "emergency", "medicate",
}

EXIT_COMMANDS = {"quit", "exit", "bye", "goodbye", "stop"}
