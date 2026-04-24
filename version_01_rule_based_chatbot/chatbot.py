# chatbot.py
# Medical Chatbot - Version A
# Contains chatbot logic — rule checking and response selection

import re

from responses import (
    responses,
    greetings,
    exit_commands,
    unsafe_topics,
    unsafe_response,
    fallback_response,
    greeting_response,
)


def is_whole_word(word, text):
    """
    Returns True only if 'word' appears as a whole word in 'text'.
    Fixes the bug where 'hi' matched inside 'this' or 'high'.

    Uses word boundary regex: \b marks the edge of a word.
    Example:
        is_whole_word("hi", "this")        → False  
        is_whole_word("hi", "hi there")    → True   
        is_whole_word("hi", "say hi now")  → True   
    """
    pattern = r"\b" + re.escape(word) + r"\b"
    return bool(re.search(pattern, text))


def get_response(user_input):
    """
    Takes user input as a string.
    Returns the appropriate bot response as a string.
    Returns "EXIT" as a special signal to end the conversation.
    """

    # Step 1: Clean the input
    user_input = user_input.strip().lower()

    # Step 2: Check for empty input
    if not user_input:
        return "Please type a question. I am here to help."

    # Step 3: Check for exit command (whole word)
    for cmd in exit_commands:
        if is_whole_word(cmd, user_input):
            return "EXIT"

    # Step 4: Check for greeting (whole word — fixes "hi" inside "this"/"high")
    for greet in greetings:
        if is_whole_word(greet, user_input):
            return greeting_response

    # Step 5: Check for unsafe topics (whole word)
    for topic in unsafe_topics:
        if is_whole_word(topic, user_input):
            return unsafe_response

    # Step 6: Check for known medical topics (whole word)
    for keyword, answer in responses.items():
        if is_whole_word(keyword, user_input):
            return answer

    # Step 7: Fallback — topic not recognized
    return fallback_response
