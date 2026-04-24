"""Configuration for the A++++ semantic chatbot -- Day 2 (FAISS vector store)."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.json"

# FAISS index and metadata are persisted in vector_store/ so embeddings
# do not need to be recomputed on every run.
VECTOR_STORE_DIR      = BASE_DIR / "vector_store"
FAISS_INDEX_PATH      = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH         = VECTOR_STORE_DIR / "metadata.json"

# ── Embedding model ───────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dimensional dense vectors, ~80 MB, fast, well-tested.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION  = 384   # must match the model output dimension

# ── Retrieval ─────────────────────────────────────────────────────────────────
# top_k: how many candidate results to retrieve from FAISS before filtering.
# Retrieving multiple candidates is the foundation of RAG --
# the LLM (Day 3+) will receive all top_k results as context.
TOP_K = 3

# Minimum cosine similarity to accept a result as relevant.
# FAISS returns L2 distance internally; we convert to cosine similarity
# by normalising vectors before indexing (see vector_store.py).
SIMILARITY_THRESHOLD = 0.35

# ── Deduplication ─────────────────────────────────────────────────────────────
# If two retrieved results share the same topic AND their scores differ by
# less than this amount, treat them as duplicates and keep only the better one.
DEDUP_SCORE_DELTA = 0.05

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
    "Medical Information Chatbot (A++++ -- Day 2)\n"
    "Powered by FAISS vector index + sentence-transformer embeddings.\n"
    "Ask about fever, headache, cough, cold, or dehydration.\n"
    "Type 'quit', 'exit', or 'bye' to stop."
)

# ── Safety ────────────────────────────────────────────────────────────────────
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
