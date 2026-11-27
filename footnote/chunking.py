"""Group small segments into retrieval-sized chunks."""
import os
import json
from pathlib import Path

from footnote.utils import course_dirs, load_course

TARGET_WORDS = int(os.getenv("FOOTNOTE_CHUNK_WORDS", "150"))  # about a minute of speech  
OVERLAP_SEGMENTS = 2  # repeated at the start of the next chunk

def n_words(segments: list[dict]) -> int:
    return sum(len(s["text"].split()) for s in segments)

def group(segments: list[dict], target_words: int = TARGET_WORDS, overlap: int = OVERLAP_SEGMENTS) -> list[list[dict]]:
    """Group segments into chunks of about `target_words` words, with some overlap."""
    groups, current, fresh = [], [], 0
    for seg in segments:
        current.append(seg)
        fresh += 1
        if n_words(current) >= target_words:
            groups.append(current)
            current = current[-overlap:] if overlap else []
            fresh = 0
    if fresh:  # leftover segments not yet in any chunk
        groups.append(current)
    return groups

def chunks_for_doc(doc: dict, course: str, document: str | None = None, 
                   document_title: str | None = None) -> list[dict]:

    kind = doc.get("kind", "text")
    # A figure description is self-contained: one page, one chunk. Everything else is grouped.
    groups = [[s] for s in doc["segments"]] if kind == "figure" else group(doc["segments"])
    # doc_id is a content hash, so the same file in two lectures would give the same ids:
    # put where the chunk comes from in front of it.
    prefix = "/".join(part for part in (course, doc["lecture"], document) if part)
    chunks = []
    for n, segs in enumerate(groups):
        first, last = segs[0], segs[-1]
        meta = {
            "course": course,
            "doc_id": doc["doc_id"],
            "lecture": doc["lecture"],
            "document": document,              # notes only: d01, d02, ...
            "document_title": document_title,  # notes only: its title in course.json
            "source_type": doc["source_type"],
            "kind": kind,
            "start": first["start"],        # video only
            "end": last["end"],             # video only
            "page_start": first["page"],    # notes only
            "page_end": last["page"],       # notes only
        }
        chunks.append({
            "id": f"{prefix}/{doc['doc_id']}-c{n:04d}",
            "text": " ".join(s["text"] for s in segs),
            "metadata": {k: v for k, v in meta.items() if v is not None},
        })
    return chunks

def processed_files(course_dir: Path) -> list[Path]:
    """The JSON files that hold segments, listed by name (convert.json lives there too).

    Sorted like the old processed/*/*.json listing, so the index is built in the same order.
    """
    processed = course_dir / "processed"
    paths = [*processed.glob("*/transcript.json"),
             *processed.glob("*/docs/*/notes.json"),
             *processed.glob("*/docs/*/figures.json")]
    return sorted(paths, key=lambda p: (p.relative_to(processed).parts[0], p.name, p.parent.name))


def load_all_chunks() -> list[dict]:
    chunks = []
    for course_dir in course_dirs():
        lectures = load_course(course_dir)["lectures"]
        for path in processed_files(course_dir):
            doc = json.loads(path.read_text())
            document = None if path.name == "transcript.json" else path.parent.name
            title = lectures.get(doc["lecture"], {}).get("docs", {}).get(document, {}).get("title")
            chunks.extend(chunks_for_doc(doc, course_dir.name, document, title))
    return chunks

