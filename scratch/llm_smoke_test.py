"""Check the Gemini API key works and list the Flash models you can use.

Usage:
    python scratch/llm_smoke_test.py                 # list models
    python scratch/llm_smoke_test.py <model-name>    # also send a test question
"""

import sys

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

flash = [m.name for m in client.models.list() if "flash" in m.name]
print("Flash models avialable:", flash)
for name in flash:
    print(" ", name)

if len(sys.argv) > 1:
    reply = client.models.generate_content(
        model=sys.argv[1],
        contents="in one sentence, what is binary search?",
    )

    print("\nReply:", reply.text)