"""Evaluate Footnote on a labelled question set.

Usage (from the Footnote folder):
    python -m footnote.evaluate retrieval   # search only: fast, free, no LLM calls
    python -m footnote.evaluate full        # full Q&A: one LLM call per question
"""

import json
import sys
import re
import time
import statistics
import unicodedata
from datetime import datetime
from pathlib import Path

from footnote import llm
from footnote.chunking import TARGET_WORDS
from footnote.index import EMBED_MODEL
from footnote.qa import K_NOTES, K_VIDEO, K_FIGURES, REFUSAL, answer, retrieve
from footnote.utils import write_json_atomic

QUESTIONS = Path("eval/questions.jsonl")
RESULTS_DIR = Path("eval/results")
PAUSE_SECONDS = 6  # between LLM calls, to stay under free-tier per-minute limits

LATEX_MARKUP = re.compile(r"[$\\{}_^]")  # $\Theta(n \log_2 n)$ -> theta(n log2 n)

def load_questions() -> list[dict]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]

def norm(text: str)-> str:
    """Lowercase, normalize Unicode and curly apostrophes, collapse spaces."""
    text = unicodedata.normalize("NFKC", text).replace("\u2019", "'").lower()
    text = LATEX_MARKUP.sub("", text)
    return " ".join(text.split())

def matches(meta: dict, gold: dict, course: str)-> bool:
    """Does one retrieved chunk overlap one gold location?"""
    if meta.get("course") != course or meta["lecture"] != gold["lecture"]:
        return False
    if meta["source_type"] != gold["type"]:
        return False
    if gold["type"] == "video":
        return meta["start"] < gold["end"] and meta["end"] > gold["start"]
    return meta["page_start"] <= gold["page"] <= meta["page_end"]

def is_relevant(hit: dict, q:dict)-> bool:
    # For figure questions the answer exists only in a drawing, so only a figure chunk counts
    if q.get("needs_figures") and hit["metadata"].get("kind") != "figure":
        return False
    return any(matches(hit["metadata"], g, q["course"]) for g in q["gold"])

def first_relevant_rank(hits: list[dict], q: dict)-> int | None:
    for rank, hit in enumerate(hits, 1):
        if is_relevant(hit, q):
            return rank
    return None

def keywords_ok(text: str, groups: list[list[str]])-> bool:
    """Every group needs at least one of its keywords somewhere in the answer."""
    t = norm(text)
    return all(any(norm(k) in t for k in group) for group in groups)

def run_retrieval(questions: list[dict]) -> list[dict]:
    rows = []
    for q in questions:
        if not q["answerable"]:
            continue # out-of-scope questions have nothing to find
        hits = retrieve(q["question"], q.get("course"))
        rows.append({"id": q["id"], "answerable": True,
                     "needs_figures": q.get("needs_figures", False),
                     "rank": first_relevant_rank(hits, q)})
    return rows

def run_full(questions: list[dict]) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        if i>1 and llm.RATE_LIMITED:
            time.sleep(PAUSE_SECONDS)  # avoid exceeding free-tier per-minute limits
        start = time.perf_counter()
        try:
            result = answer(q["question"], q["course"])
        except llm.QuotaExhaustedError as e:
            print(f"  STOPPED at {q['id']}: {e}")
            break
        except Exception as e:
            rows.append({"id": q["id"], "answerable": q["answerable"], "error": repr(e)})
            print(f"  [{i}/{len(questions)}] {q['id']}: ERROR {e!r}")
            continue

        hits, text = result["hits"], result["answer"]
        row = {"id": q["id"], "answerable": q["answerable"],
               "needs_figures": q.get("needs_figures", False),
               "refused": norm(REFUSAL) in norm(text),
               "invalid_citations": len(result["invalid"]),
               "answer": text,
               "seconds": round(time.perf_counter() - start, 1)}
        if q["answerable"]:
            row["rank"] = first_relevant_rank(hits, q)
            row["keywords_ok"] = keywords_ok(text, q["keywords"])
            row["cited_gold"] = any(is_relevant(hits[n - 1], q) for n, _ in result["sources"])
        rows.append(row)
        print(f"  [{i}/{len(questions)}] {q['id']}: {'refused' if row['refused'] else 'answered'}")
    return rows

def pct(values: list[bool]) -> str:
    if not values:
        return "n/a"
    return f"{sum(values)}/{len(values)} ({sum(values) / len(values):.0%})"

def summarize(rows: list[dict], full: bool) -> dict:
    rows = [r for r in rows if "error" not in r]  # failed calls count neither way
    text = [r for r in rows if r["answerable"] and not r["needs_figures"]]
    figs = [r for r in rows if r["answerable"] and r["needs_figures"]]
    summary = {
        "retrieval_hit_rate": pct([r["rank"] is not None for r in text]),
        "mrr": round(sum(1 / r["rank"] for r in text if r["rank"]) / len(text), 3) if text else None,
        "figure_retrieval_hit_rate": pct([r["rank"] is not None for r in figs]),
    }
    if full:
        out_of_scope = [r for r in rows if not r["answerable"]]
        summary |= {
            "answer_keywords_ok": pct([r["keywords_ok"] for r in text]),
            "cited_a_correct_source": pct([r["cited_gold"] for r in text]),
            "wrongly_refused": pct([r["refused"] for r in text]),
            "correctly_refused_out_of_scope": pct([r["refused"] for r in out_of_scope]),
            "figure_answers_ok": pct([r["keywords_ok"] for r in figs]),
            "answers_with_invalid_citations": pct([r["invalid_citations"] > 0 for r in rows]),
            "median_seconds_per_answer": statistics.median(r["seconds"] for r in rows) if rows else None,
        }
    return summary

def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    only = set(sys.argv[2:])  # optional: specific question ids, e.g. q24 q25
    if mode not in ("retrieval", "full"):
        print(__doc__)
        return
    full = mode == "full"
    questions = load_questions()
    if only:
        questions = [q for q in questions if q["id"] in only]
    rows = run_full(questions) if full else run_retrieval(questions)
    summary = summarize(rows, full)

    print(f"\n{mode} evaluation, {len(questions)} questions")
    for key, value in summary.items():
        print(f"  {key:34s} {value}")
    ok = [r for r in rows if "error" not in r]
    print("  retrieval misses:", [r["id"] for r in rows if r.get("rank", 0) is None] or "none")
    if full:
        print("  keyword fails:   ", [r["id"] for r in ok if r.get("keywords_ok") is False] or "none")
        print("  refusal mistakes:", [r["id"] for r in ok
                                      if r["refused"] != (not r["answerable"])] or "none")
        print("  errors (excluded):", [r["id"] for r in rows if "error" in r] or "none")

    suffix = "-subset" if only else ""
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{mode}{suffix}.json"
    write_json_atomic(out, {
        "mode": mode,
        "settings": {"llm_model": llm.MODEL if full else None,
                     "llm_provider": llm.PROVIDER if full else None,
                      "embed_model": EMBED_MODEL,
                     "target_words": TARGET_WORDS, "k_video": K_VIDEO, "k_notes": K_NOTES, "k_figures": K_FIGURES},
        "summary": summary,
        "rows": rows,
    })
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()