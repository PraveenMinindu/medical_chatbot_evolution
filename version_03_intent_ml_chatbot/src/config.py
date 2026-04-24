from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

INTENTS_PATH = DATA_DIR / "intents.json"
MODEL_PATH = MODELS_DIR / "intent_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"

# Raised from 0.15 back to 0.35 now that we have more training patterns.
# More patterns per intent = model is more confident = higher threshold is safe.
# A threshold of 0.15 on an 11-class problem is barely above random (1/11 = 0.09).
CONFIDENCE_THRESHOLD = 0.20

SUPPORTED_TOPICS = [
    "fever",
    "headache",
    "cough",
    "cold",
    "dehydration",
]

EXIT_COMMANDS = {"quit", "exit", "bye"}

FALLBACK_MESSAGE = (
    "Sorry, I can only provide basic educational information about fever, "
    "headache, cough, cold, and dehydration."
)

SAFE_SCOPE_MESSAGE = (
    "I can only provide general educational information. Please consult a "
    "qualified healthcare professional."
)

URGENT_MESSAGE = (
    "This may require professional medical attention. Please contact a "
    "qualified healthcare professional."
)

DISCLAIMER_MESSAGE = (
    "I provide general educational information only and I am not a doctor."
)
