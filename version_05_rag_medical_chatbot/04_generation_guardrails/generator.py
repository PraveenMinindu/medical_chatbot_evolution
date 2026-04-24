"""LLM generation layer with provider abstraction.

WHY PROVIDER ABSTRACTION MATTERS:
    LLM providers change rapidly. OpenAI releases new models, prices change,
    local models improve, new providers emerge. If every part of the code
    talks directly to the OpenAI client, switching providers means changing
    many files.

    The generator.py module is the only place that knows which provider is
    in use. Everything else calls generate() and gets back a GenerationResult.
    To switch from OpenAI to Ollama, you change one environment variable.

SUPPORTED PROVIDERS:
    1. mock    -- Returns a structured placeholder. No API needed. Used for
                  testing the pipeline without any LLM configured.

    2. openai  -- Uses the OpenAI API (or any OpenAI-compatible endpoint).
                  Requires: OPENAI_API_KEY environment variable.
                  Model: configurable via OPENAI_MODEL (default: gpt-4o-mini).
                  Also works with: Groq, Together AI, Anthropic (with adapter),
                  any provider that implements the OpenAI messages API.

    3. ollama  -- Uses a locally running Ollama server.
                  Requires: ollama running at localhost:11434 (or OLLAMA_BASE_URL).
                  Model: configurable via OLLAMA_MODEL (default: llama3.2).
                  Setup: `ollama pull llama3.2` then `ollama serve`.
                  No API key needed. Runs entirely on your machine.

HOW TO CONFIGURE:
    Set environment variables before running:

    For OpenAI:
        export LLM_PROVIDER=openai
        export OPENAI_API_KEY=sk-...
        export OPENAI_MODEL=gpt-4o-mini

    For Ollama (local):
        export LLM_PROVIDER=ollama
        export OLLAMA_MODEL=llama3.2
        # Make sure `ollama serve` is running

    For testing without any LLM:
        export LLM_PROVIDER=mock
        (or just don't set it -- mock is the default)
"""

from __future__ import annotations

import json

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
    """Route to the configured LLM provider and return a GenerationResult.

    This is the single public function of this module. All callers use this
    function regardless of which provider is configured.

    Args:
        system_prompt: The role/instructions prompt (sent as "system" role).
        user_message:  The context + question prompt (sent as "user" role).

    Returns:
        GenerationResult with answer text and provider metadata.
    """
    provider = LLM_PROVIDER.lower().strip()

    if provider == "openai":
        return _call_openai(system_prompt, user_message)
    elif provider == "ollama":
        return _call_ollama(system_prompt, user_message)
    elif provider == "mock":
        return _mock_response(user_message)
    else:
        # Unknown provider -- fall back to mock with a warning
        print(f"Warning: Unknown LLM_PROVIDER '{provider}'. Using mock.")
        return _mock_response(user_message)


# ── OpenAI provider ───────────────────────────────────────────────────────────

def _call_openai(system_prompt: str, user_message: str) -> GenerationResult:
    """Call the OpenAI API (or any OpenAI-compatible endpoint).

    Uses the openai Python library which handles auth, retries, and timeouts.
    The base_url can be changed to use Groq, Together, or other providers.
    """
    try:
        from openai import OpenAI

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Export it as an environment variable or use LLM_PROVIDER=mock."
            )

        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )

        answer = response.choices[0].message.content.strip()

        return GenerationResult(
            answer        = answer,
            provider      = "openai",
            model         = OPENAI_MODEL,
            used_fallback = False,
        )

    except ImportError:
        print("openai library not installed. Run: pip install openai")
        return _fallback_result("openai", OPENAI_MODEL)
    except Exception as exc:
        print(f"OpenAI API error: {exc}")
        return _fallback_result("openai", OPENAI_MODEL)


# ── Ollama provider ───────────────────────────────────────────────────────────

def _call_ollama(system_prompt: str, user_message: str) -> GenerationResult:
    """Call a locally running Ollama server.

    Ollama exposes an OpenAI-compatible /api/chat endpoint.
    We use the requests library directly to avoid adding the openai dependency
    for users who only want to use a local model.

    Setup:
        1. Install Ollama: https://ollama.com
        2. Pull a model: ollama pull llama3.2
        3. Start server: ollama serve
    """
    try:
        import requests

        url = f"{OLLAMA_BASE_URL}/api/chat"

        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
            "options": {
                "temperature":  LLM_TEMPERATURE,
                "num_predict":  LLM_MAX_TOKENS,
            },
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        data   = response.json()
        answer = data["message"]["content"].strip()

        return GenerationResult(
            answer        = answer,
            provider      = "ollama",
            model         = OLLAMA_MODEL,
            used_fallback = False,
        )

    except Exception as exc:
        print(f"Ollama error: {exc}")
        print(f"Is Ollama running at {OLLAMA_BASE_URL}? Run: ollama serve")
        return _fallback_result("ollama", OLLAMA_MODEL)


# ── Mock provider ─────────────────────────────────────────────────────────────

def _mock_response(user_message: str) -> GenerationResult:
    """Return a structured mock response for testing without a real LLM.

    The mock response clearly identifies itself as a mock so the developer
    knows no real LLM was called. It shows the prompt structure was built
    correctly even without an API call.

    The mock response is useful for:
      - Testing the full pipeline (safety -> retrieval -> prompt -> response)
      - Verifying the prompt was built correctly
      - Running tests in CI without requiring API keys
    """
    # Extract the question from the user_message (after "QUESTION: ")
    question = "your question"
    for line in user_message.split("\n"):
        if line.startswith("QUESTION:"):
            question = line.replace("QUESTION:", "").strip()
            break

    answer = (
        f"[MOCK RESPONSE -- no LLM configured]\n\n"
        f"The pipeline retrieved relevant context and built a grounded prompt "
        f"for the question: '{question}'\n\n"
        f"To get a real LLM-generated answer:\n"
        f"  For OpenAI: export LLM_PROVIDER=openai && export OPENAI_API_KEY=sk-...\n"
        f"  For Ollama: export LLM_PROVIDER=ollama  (then: ollama pull llama3.2)\n\n"
        f"This is general educational information only. "
        f"Please consult a healthcare professional for personal medical advice."
    )

    return GenerationResult(
        answer        = answer,
        provider      = "mock",
        model         = "mock",
        used_fallback = False,
    )


def _fallback_result(provider: str, model: str) -> GenerationResult:
    """Return a safe fallback when an LLM call fails."""
    return GenerationResult(
        answer=(
            "I was unable to generate a response at this time. "
            "Please try again. "
            "This is general educational information only. "
            "Please consult a healthcare professional for personal medical advice."
        ),
        provider      = provider,
        model         = model,
        used_fallback = True,
    )
