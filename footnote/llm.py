"""The only module that talks to an LLM provider. To switch providers, change this file."""
import os 
import random
import time


from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
MODEL = os.getenv("LLM_MODEL", "")
RETRY_CODES = {429, 500, 503}  # rate limited, or a temporary server problem
MAX_ATTEMPTS = 5  

_client = None

class EmptyResponseError(RuntimeError):
    """The API call succeeded, but the model returned no text."""

class QuotaExhaustedError(RuntimeError):
    """The provider's daily quota is used up; retrying now won't help."""

def _gemini_client():
    """Create the Gemini client once, on first use."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()
    return _client

def _why_empty(response) -> str:
    """Explain an empty reply: the model's finish reason, or why the prompt was blocked."""
    if response.candidates:
        return f"finish_reason={response.candidates[0].finish_reason}"
    feedback = response.prompt_feedback
    if feedback and feedback.block_reason:
        return f"prompt blocked: {feedback.block_reason}"
    return "no candidates returned"

def generate(prompt: str, system: str | None = None) -> str:
    """Send a text prompt and return the model's text reply."""
    if PROVIDER != "gemini":
        raise NotImplementedError(f"LLM_PROVIDER={PROVIDER!r} is not supported yet")
    if not MODEL:
        raise RuntimeError("Set LLM_MODEL in .env (pick one with scratch/llm_smoke_test.py)")

    from google.genai import errors, types

    config = types.GenerateContentConfig(system_instruction=system, temperature=0.2)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _gemini_client().models.generate_content(
                model=MODEL, contents=prompt, config=config
            )
        except errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                raise QuotaExhaustedError(
                    f"Daily quota used up for {MODEL}. Wait for the reset, switch "
                    f"LLM_MODEL, or enable billing. Usage: https://ai.dev/rate-limit"
                ) from e  
            if e.code not in RETRY_CODES or attempt == MAX_ATTEMPTS:
                raise
            wait = min(60, 5 * 2 ** (attempt - 1)) + random.uniform(0, 2)
            print(f"  LLM error {e.code} (attempt {attempt}/{MAX_ATTEMPTS}): {str(e)[:160]}")
            print(f"  retrying in {wait:.0f}s ...")
            time.sleep(wait)
            continue
        text = response.text
        if not text:
            raise EmptyResponseError(f"{MODEL} returned no text ({_why_empty(response)})")
        return text
    raise RuntimeError("retry loop ended without returning or raising")  # should never happen
