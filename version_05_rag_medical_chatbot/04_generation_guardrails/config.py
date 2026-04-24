"""Central configuration for A++++ Medical Chatbot -- Day 4 (Full RAG)."""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent.parent
RAW_KNOWLEDGE_PATH = BASE_DIR / "data" / "raw" / "medical_knowledge.md"
CHUNKS_PATH        = BASE_DIR / "data" / "processed" / "chunks.json"
VECTOR_STORE_DIR   = BASE_DIR / "vector_store"
FAISS_INDEX_PATH   = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH      = VECTOR_STORE_DIR / "metadata.json"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION  = 384

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K                = 5
SIMILARITY_THRESHOLD = 0.30

# ── LLM provider ──────────────────────────────────────────────────────────────
# Set LLM_PROVIDER in your .env or shell environment.
#
# Supported values:
#   "openai"   -- uses OpenAI API (requires OPENAI_API_KEY)
#   "ollama"   -- uses local Ollama server (no key needed, model must be pulled)
#   "mock"     -- returns a structured placeholder (no API needed, for testing)
#
# The generator.py layer reads this and routes accordingly.
# Swap the provider without touching any other file.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")

# OpenAI settings (used when LLM_PROVIDER="openai")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Ollama settings (used when LLM_PROVIDER="ollama")
# Ollama runs locally at localhost:11434 by default.
# Pull a model first: ollama pull llama3.2
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

# LLM generation parameters
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.2"))
# Low temperature (0.1-0.3) makes the model more focused and less creative.
# For a medical information system, we want consistent, grounded answers,
# not creative variation. 0.2 is a safe starting point.

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

URGENT_PHRASES = {
    "chest pain", "cant breathe", "cannot breathe",
    "unconscious", "severe bleeding", "am i dying",
    "heart attack", "stroke",
}

EXIT_COMMANDS = {"quit", "exit", "bye", "goodbye", "stop"}

# ── Messages ──────────────────────────────────────────────────────────────────
WELCOME_MESSAGE = (
    "Medical Information Chatbot (A++++ -- Day 4 -- Full RAG)\n"
    "Retrieval-Augmented Generation with LLM grounding.\n"
    "Ask about fever, headache, cough, cold, or dehydration.\n"
    "Type 'quit' to stop."
)

SAFETY_MESSAGE = (
    "I can only provide general educational information. "
    "Please consult a qualified healthcare professional for personal medical advice."
)

URGENT_MESSAGE = (
    "This sounds like it may require immediate medical attention. "
    "Please contact a healthcare professional or emergency services right away. "
    "This chatbot cannot provide emergency guidance."
)

FALLBACK_MESSAGE = (
    "I could not find relevant information in my knowledge base for that question. "
    "I can provide general educational information about fever, headache, cough, "
    "cold, and dehydration. Please try rephrasing your question."
)

INSUFFICIENT_CONTEXT_MESSAGE = (
    "The retrieved context does not contain enough information to answer "
    "that question confidently. Please try rephrasing, or ask about one of "
    "the supported topics: fever, headache, cough, cold, or dehydration."
)
