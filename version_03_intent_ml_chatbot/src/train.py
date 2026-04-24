from __future__ import annotations

from collections import Counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .config import INTENTS_PATH, MODEL_PATH, MODELS_DIR, VECTORIZER_PATH
from .utils import clean_text, load_intents


def prepare_training_data(intents: list[dict]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []

    for intent in intents:
        tag = intent["tag"]
        for pattern in intent.get("patterns", []):
            # Patterns are example user messages that teach the model each intent.
            texts.append(clean_text(pattern))
            labels.append(tag)

    return texts, labels


def train_intent_model() -> tuple[LogisticRegression, TfidfVectorizer]:
    intents = load_intents(INTENTS_PATH)
    texts, labels = prepare_training_data(intents)

    if not texts or not labels:
        raise ValueError("Training data is empty. Please check data/intents.json.")

    print("Loaded intents:", len(intents))
    print("Training examples:", len(texts))
    print("Intent distribution:", dict(Counter(labels)))

    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    # Vectorization turns text into numbers so the classifier can learn patterns.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x_train_vectors = vectorizer.fit_transform(x_train)
    x_test_vectors = vectorizer.transform(x_test)

    # LogisticRegression is beginner-friendly and gives class probabilities.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train_vectors, y_train)

    predictions = model.predict(x_test_vectors)
    print("\nEvaluation on validation split:")
    print(classification_report(y_test, predictions, zero_division=0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")

    return model, vectorizer


if __name__ == "__main__":
    train_intent_model()
