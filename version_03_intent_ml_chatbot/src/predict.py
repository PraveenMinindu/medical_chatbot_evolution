from __future__ import annotations

from dataclasses import dataclass

import joblib

from .config import MODEL_PATH, VECTORIZER_PATH
from .utils import clean_text


@dataclass
class PredictionResult:
    tag: str
    confidence: float
    cleaned_text: str


def load_prediction_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model files were not found. Run 'python -m src.train' first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_intent(text: str) -> PredictionResult:
    model, vectorizer = load_prediction_artifacts()
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])

    predicted_tag = str(model.predict(text_vector)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(text_vector)[0]
        confidence = float(max(probabilities))
    else:
        confidence = 1.0

    return PredictionResult(
        tag=predicted_tag,
        confidence=confidence,
        cleaned_text=cleaned_text,
    )
