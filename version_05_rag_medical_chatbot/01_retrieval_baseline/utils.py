"""Utility functions: text cleaning, data loading, safety checking."""

from __future__ import annotations

import json
import re
import string
from pathlib import Path
from typing import Iterable


def clean_text(text: str) -> str:
    """Normalize text for consistent processing.

    Steps:
        1. Lowercase
        2. Remove punctuation
        3. Collapse multiple spaces
        4. Strip leading/trailing whitespace

    Note: For embedding models, light cleaning is better than aggressive
    cleaning. The model understands punctuation and sentence structure.
    We clean only to standardize, not to strip all context.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_knowledge_base(file_path: Path) -> list[dict]:
    """Load and return all entries from the knowledge base JSON file."""
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def contains_restricted_keywords(text: str, keywords: Iterable[str]) -> bool:
    """Check whether the input contains any restricted keyword as a whole word.

    Uses regex word boundaries (\\b) to avoid false positives.
    Example: 'treat' does NOT match inside 'treatment' with this approach.
    """
    cleaned = clean_text(text)
    return any(
        re.search(r"\b" + re.escape(kw) + r"\b", cleaned)
        for kw in keywords
    )
