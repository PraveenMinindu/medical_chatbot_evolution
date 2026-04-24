# chatbot.py
# Medical Chatbot - Version A+
# Core logic — keyword detection and response selection

from keywords import (
    topic_keywords,
    greetings,
    exit_words,
    unsafe_keywords,
    greeting_response,
    unsafe_response,
    fallback_response,
)
from utils import clean_text, extract_keywords, is_whole_word


def get_response(user_input):
    """
    Takes a raw user sentence.
    Cleans it, searches for keywords, and returns the best response.

    Bugs fixed vs original A+:
    - Whole word matching: 'hi' no longer fires inside 'high' or 'this'
    - Topic grouping: 'headache' and 'head' belong to the same topic,
      so only ONE response is returned per topic — no duplicates.
    """

    cleaned = clean_text(user_input)

    if not cleaned:
        return "Please type a question. I am here to help."

    # Check exit (whole word)
    if extract_keywords(cleaned, exit_words):
        return "EXIT"

    # Check greeting (whole word — 'hi' won't match 'high' anymore)
    if extract_keywords(cleaned, greetings):
        return greeting_response

    # Check unsafe topics (whole word)
    if extract_keywords(cleaned, unsafe_keywords):
        return unsafe_response

    # Search topics — one response per topic maximum
    for topic, data in topic_keywords.items():
        for keyword in data["keywords"]:
            if is_whole_word(keyword, cleaned):
                return data["response"]  # First keyword hit → return topic response

    return fallback_response


def get_all_matches(user_input):
    """
    Returns all TOPIC responses found in a sentence — one per topic, no duplicates.
    Fixes the old bug where 'headache' and 'head' both matched and gave two responses
    for the same topic.

    Example:
        "I have fever and headache"
        → fever topic matched   → 1 response
        → headache topic matched → 1 response
        → total: 2 responses (correct, no duplicate)
    """
    cleaned = clean_text(user_input)
    matched_responses = []

    for topic, data in topic_keywords.items():
        for keyword in data["keywords"]:
            if is_whole_word(keyword, cleaned):
                matched_responses.append((topic, data["response"]))
                break  # Only add this topic ONCE — move to next topic

    return matched_responses
