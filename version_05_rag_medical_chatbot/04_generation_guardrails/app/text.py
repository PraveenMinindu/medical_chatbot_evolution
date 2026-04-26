"""Text utilities: cleaning and keyword detection."""

from __future__ import annotations

import re
import string
from typing import Iterable


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def contains_whole_word(text: str, keywords: Iterable[str]) -> bool:
    """Return True if any keyword appears as a whole word in text.

    Uses \\b word boundaries so 'treat' does not match inside 'treatment'.
    """
    cleaned = clean_text(text)
    return any(
        re.search(r"\b" + re.escape(kw) + r"\b", cleaned)
        for kw in keywords
    )
