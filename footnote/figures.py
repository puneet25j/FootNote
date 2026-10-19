"""Describe the visual content of lecture-notes pages with a vision model.

Each described page becomes a "figure" segment in processed/lecXX/figures.json.

Usage (from the Footnote folder):
    python -m footnote.figures                 # every course, every lecture
    python -m footnote.figures 6.006           # one course
    python -m footnote.figures 6.006 lec03     # one lecture
"""
import re
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from footnote import llm
from footnote.utils import course_dirs, lecture_dirs, sha256_file, write_json_atomic

PROMPT_VERSION = 2   
RENDER_DPI = 150     # resolution of the page image sent to the model
PAUSE_SECONDS = 4    # between calls, to go easy on free-tier limits
BOILERPLATE_MARKER = "ocw.mit.edu/terms"
SEPARATOR = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)  # a line that is only dashes


PROMPT = """This image is one page of lecture notes or lecture slides.
Describe only what plain text extraction would miss: diagrams, figures, tables, worked
examples drawn as pictures (arrays, trees, grids, arrows, highlighted or circled values)
and math formulas. Include every number, label and value shown, and say what the figure
illustrates. Write math in LaTeX between single dollar signs.
Do not repeat ordinary paragraphs of text.

Describe each figure separately, with a line containing only --- between figures.
Start each one with a short heading naming the algorithm or idea it illustrates,
so it makes sense when read on its own. Put formulas with the figure they belong to;
formulas that belong to no figure go together in one final part.
If the page has nothing like that, reply exactly: NONE"""

def split_figures(text: str) -> list[str]:
    """One page's reply -> one piece per figure."""
    return [part.strip() for part in SEPARATOR.split(text) if part.strip()]

def page_title(page) -> str:
    """Text in the page's largest font: the title on a slide, a heading or label in notes."""
    spans = [span for block in page.get_text("dict")["blocks"]
             for line in block.get("lines", [])
             for span in line["spans"] if span["text"].strip()]
    if not spans:
        return ""
    biggest = max(span["size"] for span in spans)
    return " ".join(span["text"].strip() for span in spans if span["size"] >= biggest - 0.5)


def page_plan(doc) -> dict[int, str]:
    """For every page number: "describe", or the reason it is skipped."""
    titles = [page_title(page) for page in doc]
    plan = {}
    for i, page in enumerate(doc):
        if BOILERPLATE_MARKER in page.get_text():
            plan[i + 1] = "skip: license page"
        elif i + 1 < len(titles) and titles[i] and titles[i] == titles[i + 1]:
            plan[i + 1] = "skip: animation step (next page has the same title)"
        else:
            plan[i + 1] = "describe"
    return plan


def save(out_path: Path, info: dict, pages: list[dict]) -> int:
    segments = [
        {"id": f"{info['doc_id']}-p{p['page']:03d}-{n}", "kind": "figure",
         "start": None, "end": None, "page": p["page"], "section": None, "text": part}
        for p in pages if p["status"] == "described"
        for n, part in enumerate(split_figures(p["text"]))
    ]
    write_json_atomic(out_path, {
        **info,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages": pages,
        "segments": segments,
    })
    return len(segments)

def process_lecture(course_dir: Path, lecture_dir: Path)-> None:
    label = f"{course_dir.name}/{lecture_dir.name}"
    pdf = lecture_dir/"notes.pdf"
    if not pdf.exists():
        print(f"skip {label}: no notes.pdf")
        return

    out_path = course_dir / "processed" / lecture_dir.name / "figures.json"
    source_hash = sha256_file(pdf)
    info = {
        "doc_id": source_hash[:12] + "-fig",  # differs from notes.json, so chunk ids can't clash
        "lecture": lecture_dir.name,
        "source_type": "notes",
        "kind": "figure",
        "source_file": str(pdf),
        "source_sha256": source_hash,
        "model": llm.MODEL,
        "prompt_version": PROMPT_VERSION,
    }

    # Reuse pages finished earlier with the same PDF, model and prompt (resume after a crash)
    done = {}
    if out_path.exists():
        old = json.loads(out_path.read_text())
        if all(old.get(k) == info[k] for k in ("source_sha256", "model", "prompt_version")):
            done = {p["page"]: p for p in old["pages"] if p["status"] in ("described", "none")}

    with pymupdf.open(pdf) as doc:
        plan = page_plan(doc)
        todo = [num for num, status in plan.items() if status == "describe" and num not in done]
        
        print(f"{label}: {len(todo)} pages to describe" if todo else f"{label}: nothing new to describe")

        pages = []
        for page in doc:
            num = page.number + 1
            if plan[num] != "describe":
                pages.append({"page": num, "status": plan[num]})
            elif num in done:
                pages.append(done[num])
            else:
                png = page.get_pixmap(dpi=RENDER_DPI).tobytes("png")
                reply = llm.describe_image(png, prompt=PROMPT).strip()
                if reply.rstrip(".") == "NONE":
                    pages.append({"page": num, "status": "none"})
                else:
                    pages.append({"page": num, "status": "described", "text": reply})
                print(f"    p{num}:  {pages[-1]['status']}")
                save(out_path, info, pages)  # progress survives a crash
                time.sleep(PAUSE_SECONDS) 

    n_segments = save(out_path, info, pages)
    described = sum(p["status"] == "described" for p in pages)
    print(f"  {described} pages -> {n_segments} figure segments -> {out_path}")


def main() -> None:
    args = sys.argv[1:]  # [course] [lecture ...]
    course = args[0] if args else None
    for course_dir in course_dirs(course):
        for lecture_dir in lecture_dirs(course_dir, args[1:]):
            process_lecture(course_dir, lecture_dir)


if __name__ == "__main__":
    main()
