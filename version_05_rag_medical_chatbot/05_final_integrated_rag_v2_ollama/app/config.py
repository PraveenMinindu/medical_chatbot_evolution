"""Configuration for Part 5 final integrated RAG chatbot."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_KNOWLEDGE_PATH = BASE_DIR / "data" / "raw" / "medical_knowledge.md"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.30"))

#LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").strip().lower()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
RESTRICTED_KEYWORDS = {
    "diagnose",
    "diagnosis",
    "prescribe",
    "prescription",
    "medicine",
    "medication",
    "drug",
    "drugs",
    "tablet",
    "tablets",
    "antibiotic",
    "antibiotics",
    "dosage",
    "dose",
    "treatment",
    "treat",
    "cure",
    "surgery",
    "emergency",
    "medicate",
}

URGENT_PHRASES = {
    "chest pain",
    "cant breathe",
    "cannot breathe",
    "unconscious",
    "severe bleeding",
    "am i dying",
    "heart attack",
    "stroke",
}

BLOCKED_OUTPUT_PHRASES = {
    "you should take",
    "i recommend taking",
    "the dosage is",
    "take this medicine",
    "prescribed for",
    "you need antibiotics",
    "go to the emergency",
}

EXIT_COMMANDS = {"quit", "exit", "bye", "goodbye", "stop"}

WELCOME_MESSAGE = (
    "Medical Information Chatbot (A++++ Part 5 - Final Integrated RAG)\n"
    "Safety + Retrieval + Prompting + LLM + API-ready responses.\n"
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
    "Please try rephrasing or ask about fever, headache, cough, cold, or dehydration."
)
