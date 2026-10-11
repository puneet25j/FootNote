"""Group small segments into retrieval-sized chunks."""
import json
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
TARGET_WORDS = 150  # about a minute of lecture speech
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

def chunks_for_doc(doc: dict) -> list[dict]:
    chunks = []
    for n, segs in enumerate(group(doc["segments"])):
        first, last = segs[0], segs[-1]
        meta = {
            "doc_id": doc["doc_id"],
            "lecture": doc["lecture"],
            "source_type": doc["source_type"],
            "start": first["start"],        # video only
            "end": last["end"],             # video only
            "page_start": first["page"],    # notes only
            "page_end": last["page"],       # notes only
        }
        chunks.append({
            "id": f"{doc['doc_id']}-c{n:04d}",
            "text": " ".join(s["text"] for s in segs),
            "metadata": {k: v for k, v in meta.items() if v is not None},
        })
    return chunks

def load_all_chunks() -> list[dict]:
    chunks = []
    for path in sorted(PROCESSED_DIR.glob("*/*.json")):
        chunks.extend(chunks_for_doc(json.loads(path.read_text())))
    return chunks

