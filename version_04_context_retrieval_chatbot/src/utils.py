"""Utility helpers for text cleaning and data loading."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Iterable


def preprocess_text(text: str) -> str:
    """Normalize user and knowledge-base text for retrieval."""
    lowered_text = text.lower()
    no_punctuation = lowered_text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", no_punctuation).strip()


def load_knowledge_base(file_path: Path) -> list[dict[str, str]]:
    """Load the knowledge base from JSON."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def contains_restricted_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Detect requests that fall outside the safe educational scope."""
    cleaned_text = preprocess_text(text)
    return any(keyword in cleaned_text for keyword in keywords)
