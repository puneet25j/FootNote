"""Answer questions from your lectures and notes, with citations.

Usage (from the Footnote folder):
    python -m footnote.qa what is a peak in an array                  # all courses
    python -m footnote.qa --course 6.006 what is a peak in an array   # one course
"""
import argparse
import re
from pylatexenc.latex2text import LatexNodes2Text

from footnote import llm
from footnote.index import cite, make_where, search
from footnote.utils import COURSES_DIR, course_dirs, load_course

K_VIDEO = 4
K_NOTES = 2
K_FIGURES = 3
CITATION = re.compile(r"(?<!\w)\[(\d+(?:\s*,\s*\d+)*)\]")  # [3], [1, 4], but not a[0]
MATH = re.compile(r"\$[^$]*\$")  # matches $...$ math blocks

REFUSAL = "I couldn't find this in the lectures or notes."

SYSTEM_TEMPLATE = r"""You are Footnote, a study assistant{scope}.
Answer ONLY using the numbered sources provided; they come from the student's lecture recordings and notes.
Each source is labelled with its course and lecture. Sources labelled "figure" are written
descriptions of diagrams or slides; use them as evidence like any other source.
Cite sources inline by number, like [1] or [2][4]. Every factual sentence needs a citation.
If the sources contain nothing relevant to the question, reply exactly:
"{refusal}"
If they answer only part of the question, answer that part and briefly say what is missing.
Write math in LaTeX between single dollar signs, e.g. $\Theta(n \log n)$.
Do not write timestamps or page numbers yourself; the source numbers are enough."""

def system_prompt(course: str | None) -> str:
    if course:
        name = load_course(COURSES_DIR / course)["name"]
        return SYSTEM_TEMPLATE.format(scope=f" for the course {name}", refusal=REFUSAL)
    names = [load_course(d)["name"] for d in course_dirs()]
    scope = (" for the student's courses: " + "; ".join(names)) if names else ""
    return SYSTEM_TEMPLATE.format(scope=scope, refusal=REFUSAL)

def retrieve(question: str, course: str | None = None) -> list[dict]:
    """Best video chunks + best notes + figures chunks, merged and sorted by score."""
    hits = search(question, k=K_VIDEO, where=make_where(course=course, source_type="video"))
    hits += search(question, k=K_NOTES, where=make_where(course=course, source_type="notes", kind="text"))
    hits += search(question, k=K_FIGURES, where=make_where(course=course, kind="figure"))
    return sorted(hits, key=lambda h: h["score"], reverse=True)

def build_prompt(question: str, hits: list[dict]) -> str:
    blocks = []
    for n, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        figure = " figure" if meta.get("kind") == "figure" else ""
        label = f"{meta.get('course', '?')}, {meta['lecture']}, {meta['source_type']}{figure}"
        blocks.append(f"[{n}] ({label})\n{hit['text']}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"

def answer(question: str, course: str | None = None) -> dict:
    hits = retrieve(question, course)
    reply = llm.generate(build_prompt(question, hits), system=system_prompt(course))

    prose = MATH.sub(" ", reply)   # copy of the reply with math blanked out
    cited = {int(n) for group in CITATION.findall(prose) for n in group.split(",")}
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
    print(LatexNodes2Text().latex_to_text(result["answer"]).strip())
    if result["sources"]:
        print("\nSources:")
        for n, citation in result["sources"]:
            print(f"  [{n}] {citation}")
    if result["invalid"]:
        print(f"\nWARNING: answer cites sources that don't exist: {result['invalid']}")


if __name__ == "__main__":
    main()

