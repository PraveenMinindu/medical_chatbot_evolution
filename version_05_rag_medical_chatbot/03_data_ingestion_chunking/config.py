"""Configuration for A++++ Medical Chatbot -- Day 3 (Chunk-based RAG)."""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_KNOWLEDGE_PATH = BASE_DIR / "data" / "raw" / "medical_knowledge.md"
CHUNKS_PATH        = BASE_DIR / "data" / "processed" / "chunks.json"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH    = VECTOR_STORE_DIR / "metadata.json"

# ── Chunking ──────────────────────────────────────────────────────────────────
# CHUNK_SIZE: target length of each chunk in characters.
# 400 chars ≈ 2-3 sentences -- focused enough for precise retrieval.
CHUNK_SIZE = 400

# CHUNK_OVERLAP: characters repeated at the start of the next chunk.
# 80 chars ≈ 20% overlap -- preserves context across chunk boundaries.
CHUNK_OVERLAP = 80

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION  = 384

# ── Retrieval ─────────────────────────────────────────────────────────────────
# TOP_K chunks are retrieved and will form the LLM context window in Day 4.
TOP_K = 5

# Minimum cosine similarity to accept a chunk as relevant.
SIMILARITY_THRESHOLD = 0.30

# ── Messages ──────────────────────────────────────────────────────────────────
FALLBACK_MESSAGE = (
    "Sorry, I could not find relevant information for that question. "
    "I can provide general educational information about fever, headache, "
    "cough, cold, and dehydration. Please try rephrasing your question."
)

SAFETY_MESSAGE = (
    "I can only provide general educational information. "
    "Please consult a qualified healthcare professional for personal medical advice."
)

WELCOME_MESSAGE = (
    "Medical Information Chatbot (A++++ -- Day 3)\n"
    "Chunk-based retrieval from FAISS vector index.\n"
    "Ask about fever, headache, cough, cold, or dehydration.\n"
    "Type 'quit', 'exit', or 'bye' to stop.\n"
    "Tip: Run 'python scripts/ingest_data.py' first if the index is missing."
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
