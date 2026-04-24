# utils.py
# Medical Chatbot - Version A+
# Text cleaning utilities — this is the start of NLP thinking

import re


def clean_text(text):
    """
    Cleans raw user input for keyword matching.

    Steps:
    1. Convert to lowercase         → 'Fever' and 'fever' treated the same
    2. Remove punctuation           → "fever?" becomes "fever"
    3. Collapse extra whitespace    → "I  have   fever" becomes "I have fever"
    4. Strip leading/trailing space → "  hello  " becomes "hello"
    """

    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def is_whole_word(word, text):
    """
    Returns True only if 'word' appears as a whole word in 'text'.
    Fixes the bug where 'hi' matched inside 'this' or 'high'.

    Uses word boundary (\b) so partial substring matches are rejected.
    Example:
        is_whole_word("hi", "high temperature") → False  ✅ fixed
        is_whole_word("hi", "hi there")         → True   ✅
    """
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, text))


def extract_keywords(text, keyword_list):
    """
    Checks which keywords from keyword_list appear as whole words in text.
    Returns a list of matched keywords.
    Uses whole-word matching to avoid false positives like 'hi' in 'high'.
    """
    found = []
    for keyword in keyword_list:
        if is_whole_word(keyword, text):
            found.append(keyword)
    return found
