"""The only module that talks to an LLM provider. To switch providers, change this file."""
import os 
import time


from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
MODEL = os.getenv("LLM_MODEL", "")
RETRY_CODES = {429, 500, 503}  # rate limited, or a temporary server problem

_client = None

def _gemini_client():
    """Create the Gemini client once, on first use."""
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()
    return _client

def generate(prompt: str, system: str | None=None, retries: int = 3) -> str:
    """Send a text prompt and return the model's text reply."""
    if PROVIDER != "gemini":
        raise NotImplementedError(f"LLM_PROVIDER={PROVIDER!r} is not supported yet")
    if not MODEL: 
        raise RuntimeError("Set LLM_MODEL in .env (pick one with scratch/llm_smoke_test.py)")

    from google.genai import errors, types

    config = types.GenerateContentConfig(system_instruction=system, temperature=0.2)
    for attempt in range (retries):
        try:
            response = _gemini_client().models.generate_content(
                model=MODEL,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except errors.APIError as e:
            if e.code not in RETRY_CODES or attempt == retries - 1:
                raise
            wait = 5*2 ** attempt
            print(f"LLM API error {e.code}, retrying in {wait}s...")
            time.sleep(wait)
    return ""
