"""Answer questions from the lectures, with citations.

Usage (from the Footnote folder):
    python -m footnote.qa what is a peak in an array
"""
import re
import sys

from footnote import llm
from footnote.index import cite, search

K_VIDEO = 4
K_NOTES = 2
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # matches [3] and [1, 4]

SYSTEM = """You are Footnote, a study assistant for MIT 6.006 (Introduction to Algorithms).
Answer ONLY using the numbered sources provided.
Cite sources inline by number, like [1] or [2][4]. Every factual sentence needs a citation.
If the sources do not contain the answer, reply exactly:
"I couldn't find this in the lectures or notes."
Do not write timestamps or page numbers yourself; the source numbers are enough."""

def retrieve(question: str) -> list[dict]:
    """Best video chunks + best notes chunks, merged and sorted by score."""
    hits = search(question, k=K_VIDEO, where={"source_type": "video"})
    hits += search(question, k=K_NOTES, where={"source_type": "notes"})
    return sorted(hits, key=lambda h: h["score"], reverse=True)

def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for n, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        label = f"{meta['lecture']}, {meta['source_type']}"
        blocks.append(f"[{n}] ({label})\n{hit['text']}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

def answer(question: str) -> dict:
    hits = retrieve(question)
    reply = llm.generate(build_prompt(question, hits), system=SYSTEM)

    cited = {int(n) for group in CITATION.findall(reply) for n in group.split(",")}
    valid = sorted(n for n in cited if 1 <= n <= len(hits))
    invalid = sorted(n for n in cited if not 1 <= n <= len(hits))
    return {
        "answer": reply,
        "sources": [(n, cite(hits[n - 1]["metadata"])) for n in valid],
        "invalid": invalid,
        "hits": hits,
    }

def main() -> None:
    question = " ".join(sys.argv[1:])
    if not question:
        print(__doc__)
        return
    result = answer(question)
    print(result["answer"].strip())
    if result["sources"]:
        print("\nSources:")
        for n, citation in result["sources"]:
            print(f"  [{n}] {citation}")
    if result["invalid"]:
        print(f"\nWARNING: answer cites sources that don't exist: {result['invalid']}")


if __name__ == "__main__":
    main()

