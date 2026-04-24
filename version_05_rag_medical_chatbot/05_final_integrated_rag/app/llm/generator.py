"""LLM generation provider abstraction."""

from __future__ import annotations

from app.config import (
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)
from app.schemas import GenerationResult


def generate(system_prompt: str, user_message: str) -> GenerationResult:
    if LLM_PROVIDER == "openai":
        return _call_openai(system_prompt, user_message)
    if LLM_PROVIDER == "ollama":
        return _call_ollama(system_prompt, user_message)
    return _mock_response(user_message)


def _call_openai(system_prompt: str, user_message: str) -> GenerationResult:
    try:
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
        answer = (response.choices[0].message.content or "").strip()
        return GenerationResult(answer=answer, provider="openai", model=OPENAI_MODEL, used_fallback=False)
    except Exception:
        return _fallback_result("openai", OPENAI_MODEL)


def _call_ollama(system_prompt: str, user_message: str) -> GenerationResult:
    try:
        import requests

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {"temperature": LLM_TEMPERATURE, "num_predict": LLM_MAX_TOKENS},
            },
            timeout=60,
        )
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        return GenerationResult(answer=answer, provider="ollama", model=OLLAMA_MODEL, used_fallback=False)
    except Exception:
        return _fallback_result("ollama", OLLAMA_MODEL)


def _mock_response(user_message: str) -> GenerationResult:
    question = "your question"
    for line in user_message.splitlines():
        if line.startswith("QUESTION:"):
            question = line.replace("QUESTION:", "").strip()
            break
    answer = (
        "[MOCK RESPONSE]\n"
        f"Grounded prompt prepared for question: '{question}'.\n"
        "Set LLM_PROVIDER=openai or LLM_PROVIDER=ollama for real generation.\n"
        "This is general educational information only. Please consult a healthcare professional for personal medical advice."
    )
    return GenerationResult(answer=answer, provider="mock", model="mock", used_fallback=False)


def _fallback_result(provider: str, model: str) -> GenerationResult:
    return GenerationResult(
        answer=(
            "I was unable to generate a response at this time. Please try again. "
            "This is general educational information only. Please consult a healthcare professional for personal medical advice."
        ),
        provider=provider,
        model=model,
        used_fallback=True,
    )

