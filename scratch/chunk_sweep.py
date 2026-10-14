"""Compare chunk sizes using the retrieval evaluation (search only, no LLM calls).

Usage (from the Footnote folder):  python -m scratch.chunk_sweep
"""
import json
import os 
import statistics
import subprocess
import sys 
import time
from pathlib import Path

from footnote.chunking import group
from footnote.utils import course_dirs  

SIZES = [75, 100, 150, 250, 400]
RESULTS_DIR = Path("eval/results")

def chunk_stats(size: int) -> tuple[int, float]:
    """Number of chunks, and the median time span of video chunks in seconds."""
    count , spans = 0, []
    for course_dir in course_dirs():
        for path in sorted((course_dir / "processed").glob("*/*.json")):
            doc = json.loads(path.read_text())
            groups = group(doc["segments"], target_words=size)
            count += len(groups)
            if doc["source_type"] == "video":
                spans += [g[-1]["end"] - g[0]["start"] for g in groups]
    return count, statistics.median(spans)

def run(*args: str, size: int | None = None) -> None:
    """Run `python -m <args>` as a separate process, optionally with a chunk size."""
    env = dict(os.environ)
    if size is not None:
        env["FOOTNOTE_CHUNK_WORDS"] = str(size)
    subprocess.run([sys.executable, "-m", *args], env=env, check=True)


def main() -> None:
    started = time.time()
    for size in SIZES:
        print(f"\n=== chunk size: {size} words ===")
        run("footnote.index", "build", size=size)
        run("footnote.evaluate", "retrieval", size=size)

    print("\n=== restoring the default index ===")
    run("footnote.index", "build")

    rows = []
    for path in sorted(RESULTS_DIR.glob("*-retrieval.json")):
        if path.stat().st_mtime < started:
            continue  # results from before this sweep
        result = json.loads(path.read_text())
        size = result["settings"]["target_words"]
        n_chunks, span = chunk_stats(size)
        rows.append((size, n_chunks, span, result["summary"]))

    print(f"\n{'words':>6} {'chunks':>7} {'video span':>11}  {'hit rate':<14} {'MRR':>6}")
    for size, n_chunks, span, summary in sorted(rows):
        print(f"{size:>6} {n_chunks:>7} {span:>9.0f}s  "
              f"{summary['retrieval_hit_rate']:<14} {summary['mrr']:>6}")


if __name__ == "__main__":
    main()