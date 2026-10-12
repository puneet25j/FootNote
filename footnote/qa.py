"""Answer questions from your lectures and notes, with citations.

Usage (from the Footnote folder):
    python -m footnote.qa what is a peak in an array                  # all courses
    python -m footnote.qa --course 6.006 what is a peak in an array   # one course
"""
import argparse
import re


from footnote import llm
from footnote.index import cite, make_where, search
from footnote.utils import COURSES_DIR, course_dirs, load_course

K_VIDEO = 4
K_NOTES = 2
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # matches [3] and [1, 4]

SYSTEM_TEMPLATE = """You are Footnote, a study assistant{scope}.
Answer ONLY using the numbered sources provided; they come from the student's lecture recordings and notes.
Each source is labelled with its course and lecture.
Cite sources inline by number, like [1] or [2][4]. Every factual sentence needs a citation.
If the sources do not contain the answer, reply exactly:
"I couldn't find this in the lectures or notes."
Do not write timestamps or page numbers yourself; the source numbers are enough."""

def system_prompt(course: str | None) -> str:
    if course:
        name = load_course(COURSES_DIR / course)["name"]
        return SYSTEM_TEMPLATE.format(scope=f" for the course {name}")
    names = [load_course(d)["name"] for d in course_dirs()]
    scope = (" for the student's courses: " + "; ".join(names)) if names else ""
    return SYSTEM_TEMPLATE.format(scope=scope)

def retrieve(question: str, course: str | None = None) -> list[dict]:
    """Best video chunks + best notes chunks, merged and sorted by score."""
    hits = search(question, k=K_VIDEO, where=make_where(course=course, source_type="video"))
    hits += search(question, k=K_NOTES, where=make_where(course=course, source_type="notes"))
    return sorted(hits, key=lambda h: h["score"], reverse=True)

def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for n, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        label = f"{meta.get('course', '?')}, {meta['lecture']}, {meta['source_type']}"
        blocks.append(f"[{n}] ({label})\n{hit['text']}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

def answer(question: str, course: str | None = None) -> dict:
    hits = retrieve(question, course)
    reply = llm.generate(build_prompt(question, hits), system=system_prompt(course))

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
    parser = argparse.ArgumentParser(description="Ask Footnote a question about your lectures.")
    parser.add_argument("question", nargs="+", help="your question (quotes optional)")
    parser.add_argument("--course", help="course folder in data/courses (default: all courses)")
    args = parser.parse_args()

    if args.course and not (COURSES_DIR / args.course).is_dir():
        available = ", ".join(d.name for d in course_dirs()) or "none"
        parser.error(f"unknown course {args.course!r} (available: {available})")

    result = answer(" ".join(args.question), args.course)
    print(result["answer"].strip())
    if result["sources"]:
        print("\nSources:")
        for n, citation in result["sources"]:
            print(f"  [{n}] {citation}")
    if result["invalid"]:
        print(f"\nWARNING: answer cites sources that don't exist: {result['invalid']}")


if __name__ == "__main__":
    main()

