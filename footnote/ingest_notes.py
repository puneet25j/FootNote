"""Extract lecture-notes PDFs into page-numbered segments (JSON).

Usage (from the Footnote folder):
    python -m footnote.ingest_notes            # every lecture in data/raw/
    python -m footnote.ingest_notes lec01      # just one
"""
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

from footnote.utils import already_done, sha256_file, write_json_atomic

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
PARSER_VERSION = 2

BOILERPLATE_MARKER = "ocw.mit.edu/terms"
HAS_WORD = re.compile(r"[A-Za-z]{3,}")

def clean_block(text: str) -> str:
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


def process_lecture(lecture_dir: Path) -> None:
    pdf = lecture_dir / "notes.pdf"
    if not pdf.exists():
        print(f"skip {lecture_dir.name}: no notes.pdf")
        return

    out_path = OUT_DIR / lecture_dir.name / "notes.json"
    source_hash = sha256_file(pdf)
    if already_done(out_path, source_hash, parser_version=PARSER_VERSION):
        print(f"skip {lecture_dir.name}: notes already extracted")
        return

    blocks, pages = extract(pdf)
    blocks = drop_repeated(blocks, len(pages))
    blocks = drop_noise(blocks)

    doc_id = source_hash[:12]
    segments = [
        {
            "id": f"{doc_id}-{i:05d}",
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
        "doc_id": doc_id,
        "lecture": lecture_dir.name,
        "source_type": "notes",
        "source_file": str(pdf),
        "source_sha256": source_hash,
        "parser_version": PARSER_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages": pages,
        "segments": segments,
    })

    print(f"{lecture_dir.name}: {len(pages)} pages, {len(segments)} segments -> {out_path}")
    for p in pages:
        print(f"    p{p['page']:>2}  chars={p['chars']:>5}  "
              f"images={p['images']}  drawings={p['drawings']}")


def main() -> None:
    names = sys.argv[1:]
    if names:
        lecture_dirs = [RAW_DIR / n for n in names]
    else:
        lecture_dirs = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    for d in lecture_dirs:
        process_lecture(d)


if __name__ == "__main__":
    main()