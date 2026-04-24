"""Configuration values for the retrieval-based medical chatbot."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_PATH = BASE_DIR / "data" / "knowledge_base.json"

SIMILARITY_THRESHOLD = 0.30

FALLBACK_MESSAGE = (
    "Sorry, I can only provide basic medical information about fever, "
    "headache, cough, cold, and dehydration."
)

SAFETY_MESSAGE = (
    "I can only provide general educational information. Please consult a "
    "healthcare professional."
)

EXIT_COMMANDS = {"quit", "exit", "bye"}

RESTRICTED_KEYWORDS = {
    "diagnose",
    "diagnosis",
    "medicine",
    "medication",
    "drug",
    "prescription",
    "prescribe",
    "tablet",
    "capsule",
    "dose",
    "dosage",
    "antibiotic",
    "treatment",
    "treat",
    "cure",
    "heal",
}
