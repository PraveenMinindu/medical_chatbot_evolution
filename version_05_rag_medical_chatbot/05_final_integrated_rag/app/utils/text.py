"""Text cleaning and whole-word keyword matching."""

from __future__ import annotations

import re
import string
from typing import Iterable


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def contains_whole_word(text: str, keywords: Iterable[str]) -> bool:
    cleaned = clean_text(text)
    return any(re.search(r"\b" + re.escape(kw) + r"\b", cleaned) for kw in keywords)

