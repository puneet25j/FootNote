"""The only module that talks to an LLM provider. To switch providers, change this file."""
import os 
import random
import time

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
MODEL = os.getenv("LLM_MODEL", "")
TEMPERATURE = 0.2                    
THINK = False   
RATE_LIMITED = PROVIDER == "gemini"   # callers pause between calls only for the hosted API

NUM_CTX = 8192       # Ollama: context window in tokens (prompt + reply); its default is smaller
MAX_OUTPUT = 2048     # Ollama: cap on reply tokens, so a model stuck repeating itself stops

RETRY_CODES = {429, 500, 503}  # rate limited, or a temporary server problem
MAX_ATTEMPTS = 5  

_client = None

# ---------- Errors ----------

class LLMError(RuntimeError):
    """A call failed in a way worth retrying later: server trouble, empty reply, local model problem."""
class LLMUnavailableError(LLMError):
    """The provider kept failing (500/503) after every retry."""

class EmptyResponseError(LLMError):
    """The call succeeded, but the model returned no text."""

class LocalModelError(LLMError):
    """Ollama isn't running, the model isn't downloaded, or the prompt/reply didn't fit."""

class QuotaExhaustedError(RuntimeError):
    """The daily quota is used up; retrying today won't help. Deliberately not an LLMError."""


def _check_settings() -> None:
    if PROVIDER not in ("gemini", "ollama"):
        raise NotImplementedError(f"LLM_PROVIDER={PROVIDER!r} is not supported (use gemini or ollama)")
    if not MODEL:
        raise RuntimeError("Set LLM_MODEL in .env (gemma-4-31b-it for gemini, gemma4:26b for ollama)")


# ---------- Gemini (hosted) ----------

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


def _gemini(contents, system: str | None = None) -> str:
    """One Gemini request with retries; return the reply text or raise."""
    from google.genai import errors, types
 
    config = types.GenerateContentConfig(system_instruction=system, temperature=TEMPERATURE)
 
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = _gemini_client().models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except errors.APIError as e:
            if e.code == 429 and "PerDay" in str(e):
                raise QuotaExhaustedError(
                    f"Daily quota used up for {MODEL}. Wait for the reset, switch "
                    f"LLM_MODEL, or enable billing. Usage: https://ai.dev/rate-limit"
                ) from e
            if e.code not in RETRY_CODES:
                raise
            if attempt == MAX_ATTEMPTS:
                raise LLMUnavailableError(f"{MODEL} still failing after {MAX_ATTEMPTS} attempts ({e.code})") from e
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

# ---------- Ollama (local) ----------

def _ollama(message: dict, system: str | None = None) -> str:
    """One request to the local Ollama server; return the reply text or raise."""
    import ollama
 
    messages = [{"role": "system", "content": system}] if system else []
    messages.append(message)
    try:
        response = ollama.chat(
            model=MODEL, messages=messages, think=THINK,
            options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX, "num_predict": MAX_OUTPUT},
        )
    except ConnectionError as e:
        raise LocalModelError("Can't reach Ollama. Open the Ollama app (or run `ollama serve`).") from e
    except ollama.ResponseError as e:
        if e.status_code == 404:
            raise LocalModelError(f"{MODEL!r} isn't downloaded. Run: ollama pull {MODEL}") from e
        raise LocalModelError(f"Ollama error {e.status_code}: {e.error}") from e
 
    used = response.prompt_eval_count or 0
    if used > NUM_CTX - MAX_OUTPUT:
        raise LocalModelError(f"Prompt used {used} of {NUM_CTX} tokens and may have been cut. Raise NUM_CTX.")
    if response.done_reason == "length":
        raise LocalModelError(f"Reply hit the {MAX_OUTPUT}-token cap (the model was probably repeating itself)")
    text = response.message.content
    if not text:
        raise EmptyResponseError(f"{MODEL} returned no text (done_reason={response.done_reason})")
    return text


# ---------- Public API: the rest of Footnote calls only these ----------

def generate(prompt: str, system: str | None = None) -> str:
    """Send a text prompt and return the model's text reply."""
    _check_settings()
    if PROVIDER == "ollama":
        return _ollama({"role": "user", "content": prompt}, system)
    return _gemini(prompt, system)


def describe_image(image: bytes, prompt: str, mime_type: str = "image/png") -> str:
    """Send one image plus an instruction and return the model's text reply."""
    _check_settings()
    if PROVIDER == "ollama":
        return _ollama({"role": "user", "content": prompt, "images": [image]})  # Ollama detects the type
    from google.genai import types
    picture = types.Part.from_bytes(data=image, mime_type=mime_type)
    return _gemini([picture, prompt])