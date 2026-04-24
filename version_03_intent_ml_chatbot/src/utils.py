import json
import random
import re
from pathlib import Path
from typing import Any


def clean_text(text: str) -> str:
    """Apply simple preprocessing so training and prediction use the same format."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_json(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_intents(file_path: Path) -> list[dict[str, Any]]:
    data = load_json(file_path)
    return data.get("intents", [])


def choose_response(responses: list[str], rng: random.Random | None = None) -> str:
    if not responses:
        return ""
    random_generator = rng or random
    return random_generator.choice(responses)


def build_response_lookup(intents: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        intent["tag"]: intent.get("responses", [])
        for intent in intents
    }


def should_use_fallback(confidence: float, threshold: float) -> bool:
    return confidence < threshold
