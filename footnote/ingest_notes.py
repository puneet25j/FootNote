"""Extract the text of each lecture document into page-numbered segments (JSON).

Usage (from the Footnote folder):
    python -m footnote.ingest_notes                 # every course, every lecture
    python -m footnote.ingest_notes 6.006           # one course
    python -m footnote.ingest_notes 6.006 lec01     # one lecture
"""
import re
import sys
import unicodedata 
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from footnote.convert import ConvertError, to_pdf
from footnote.utils import (already_done, course_dirs, document_dir, lecture_dirs,
                            lecture_documents, sha256_file, write_json_atomic)


PARSER_VERSION = 3

BOILERPLATE_MARKER = "ocw.mit.edu/terms"
HAS_WORD = re.compile(r"[A-Za-z]{3,}")

def clean_block(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # "ﬁ" -> "fi", "…" -> "..."
    text = text.replace("\u00ad", "")           # invisible "soft hyphen"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "algo-\nrithm" -> "algorithm"
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def extract(pdf_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (blocks, pages): text blocks with page numbers, plus per-page stats."""
    blocks, pages = [], []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            page_no = page.number + 1  # PyMuPDF counts from 0; people count from 1
            chars = 0
            for *_, text, _block_no, block_type in page.get_text("blocks", sort=True):
                if block_type != 0:  # 0 = text, 1 = image
                    continue
                text = clean_block(text)
                if len(text) < 3 or text.isdigit():  # stray symbols, page numbers
                    continue
                blocks.append({"page": page_no, "text": text})
                chars += len(text)
            pages.append({
                "page": page_no,
                "chars": chars,
                "images": len(page.get_images()),
                "drawings": len(page.get_drawings()),
            })
    return blocks, pages

def drop_noise(blocks: list[dict]) -> list[dict]:
    """Remove text that would only add noise to search."""
    license_pages = {b["page"] for b in blocks if BOILERPLATE_MARKER in b["text"]}
    seen, kept = set(), []
    for b in blocks:
        if b["page"] in license_pages:      # the OCW license/citation page
            continue
        if not HAS_WORD.search(b["text"]):  # figure labels like "14 13 12", "⎫ ⎪ ⎬"
            continue
        if b["text"] in seen:               # slide decks repeat text on every animation step
            continue
        seen.add(b["text"])
        kept.append(b)
    return kept


def drop_repeated(blocks: list[dict], n_pages: int) -> list[dict]:
    """Remove running headers/footers: text that appears on at least half the pages."""
    if n_pages < 3:
        return blocks
    page_count = Counter(text for _page, text in {(b["page"], b["text"]) for b in blocks})
    repeated = {text for text, count in page_count.items() if count >= n_pages / 2}
    return [b for b in blocks if b["text"] not in repeated]


def process_document(course_dir: Path, lecture_id: str, doc_id: str, src: Path) -> None:
    label = f"{course_dir.name}/{lecture_id}/{doc_id}"
    out_dir = document_dir(course_dir, lecture_id, doc_id)
    pdf = to_pdf(src, out_dir)
    out_path = out_dir / "notes.json"
    source_hash = sha256_file(pdf)
    if already_done(out_path, source_hash, parser_version=PARSER_VERSION):
        print(f"skip {label}: notes already extracted")
        return

    blocks, pages = extract(pdf)
    blocks = drop_repeated(blocks, len(pages))
    blocks = drop_noise(blocks)

    hash_id = source_hash[:12]   # content-based: chunk ids are built from it
    segments = [
        {
            "id": f"{hash_id}-{i:05d}",
            "kind": "text",
            "start": None,
            "end": None,
            "page": b["page"],
            "section": None,
            "text": b["text"],
        }
        for i, b in enumerate(blocks)
    ]

    write_json_atomic(out_path, {
        "doc_id": hash_id,
        "lecture": lecture_id,
        "source_type": "notes",
        "source_file": str(pdf),
        "source_sha256": source_hash,
        "parser_version": PARSER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages": pages,
        "segments": segments,
    })
    print(f"{label}: {len(pages)} pages, {len(segments)} segments -> {out_path}")

def process_lecture(course_dir: Path, lecture_dir: Path) -> None:
    docs = lecture_documents(course_dir, lecture_dir.name)
    if not docs:
        print(f"skip {course_dir.name}/{lecture_dir.name}: no documents")
    for doc_id, src in docs:
        label = f"{course_dir.name}/{lecture_dir.name}/{doc_id}"
        if not src.exists():
            print(f"skip {label}: {src} is missing")
            continue
        try:
            process_document(course_dir, lecture_dir.name, doc_id, src)
        except ConvertError as e:   # one bad document must not stop the others
            print(f"FAILED {label}: {e}")

def main() -> None:
    args = sys.argv[1:]                      # [course] [lecture ...]
    course = args[0] if args else None
    for course_dir in course_dirs(course):
        for lecture_dir in lecture_dirs(course_dir, args[1:]):
            process_lecture(course_dir, lecture_dir)


if __name__ == "__main__":
    main()