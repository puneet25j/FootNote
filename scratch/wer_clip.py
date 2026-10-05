"""
WER of the tiny/small clip transcripts vs the official transcript.

Usage (from the Footnote folder):  python scratch/wer_clip.py
"""
import json
import re
from pathlib import Path

import jiwer
import pymupdf

TRANSCRIPT_PDF = Path("data/raw/lec01/transcript.pdf")
OUT_DIR = Path("scratch/out")
SPEAKER_LABEL = re.compile(r"[A-Z][A-Z .'-]*:")  # PROFESSOR:, ERIK DOMANE:

def pdf_text(path: Path) -> str:
    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)

def clean_reference(text: str) -> str:
    """Remove what Whisper would never output: headers, page numbers, labels, tags."""
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.isdigit() or line.startswith("MITOCW"):
            continue
        if SPEAKER_LABEL.fullmatch(line):
            continue
        kept.append(line)
    text = " ".join(kept)
    return re.sub(r"\[[^\]]*\]", " ", text)  # [LAUGHTER], [INAUDIBLE], ...

def normalize(text: str) -> list[str]:
    """Same rules for both sides, so only real word differences remain."""
    text = text.lower().replace("'", "").replace("\u2019", "")  # don't -> dont
    text = re.sub(r"[^a-z0-9]+", " ", text)  # punctuation, dashes -> space
    return text.split()

def best_cut(ref: list[str], hyp: list[str], slack: int = 100) -> int:
    """How many reference words does the clip cover? Try nearby cut points."""
    n = len(hyp)
    candidates = range(max(1, n - slack), min(len(ref), n + slack) + 1)
    return min(candidates, key=lambda k: jiwer.wer(" ".join(ref[:k]), " ".join(hyp)))

def main() -> None:

    ref_all = normalize(clean_reference(pdf_text(TRANSCRIPT_PDF)))

    for model in ["tiny", "small"]:
        segs = json.loads((OUT_DIR / f"lec01_5min_{model}.json").read_text())
        hyp = normalize(" ".join(s["text"] for s in segs))
        ref = ref_all[: best_cut(ref_all, hyp)]

        result = jiwer.process_words(" ".join(ref), " ".join(hyp))

        print(f"{model:5s}  WER={result.wer:.1%}  ref_words={len(ref)}  "
              f"sub={result.substitutions}  del={result.deletions}  ins={result.insertions}")

        lines = []
        for chunk in result.alignments[0]:
            if chunk.type == "equal":
                continue
            r = " ".join(ref[chunk.ref_start_idx:chunk.ref_end_idx])
            h = " ".join(hyp[chunk.hyp_start_idx:chunk.hyp_end_idx])
            lines.append(f"{chunk.type:10s}  official: {r!r:35s}  {model}: {h!r}")
        (OUT_DIR / f"wer_errors_{model}.txt").write_text("\n".join(lines))
        print(f"       {len(lines)} differences -> scratch/out/wer_errors_{model}.txt")


if __name__ == "__main__":
    main()