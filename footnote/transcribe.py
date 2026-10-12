"""Transcribe lecture videos into timestamped segments (JSON).

Usage (from the Footnote folder):
    python -m footnote.transcribe                 # every course, every lecture
    python -m footnote.transcribe 6.006           # one course
    python -m footnote.transcribe 6.006 lec01     # one lecture
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from faster_whisper import WhisperModel

from footnote.utils import (already_done, course_dirs, lecture_dirs, load_course,
                            sha256_file, write_json_atomic)

MODEL_SIZE = "small"


def transcribe(media_path: Path, model_size: str, language: str) -> tuple[list[dict], float]:
    """The ONLY function that knows about Whisper. Returns (segments, duration)."""
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(media_path), language=language, beam_size=5, vad_filter=True
    )
    out = []
    for seg in segments:
        out.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        print(f"\r  progress {seg.end / info.duration:6.1%}", end="", flush=True)
    print()
    return out, info.duration


def process_lecture(course_dir: Path, lecture_dir: Path, language: str) -> None:
    label = f"{course_dir.name}/{lecture_dir.name}"
    video = lecture_dir / "video.mp4"
    if not video.exists():
        print(f"skip {lecture_dir.name}: no video.mp4")
        return

    out_path = course_dir / "processed" / lecture_dir.name / "transcript.json"
    source_hash = sha256_file(video)
    if already_done(out_path, source_hash, model=MODEL_SIZE):
        print(f"skip {label}: already transcribed")
        return

    print(f"transcribing {label} with {MODEL_SIZE} ({language}) ...")
    t0 = time.perf_counter()
    raw_segments, duration = transcribe(video, MODEL_SIZE)
    elapsed = time.perf_counter() - t0

    doc_id = source_hash[:12]
    segments = [
        {
            "id": f"{doc_id}-{i:05d}",
            "kind": "text",
            "start": s["start"],
            "end": s["end"],
            "page": None,
            "section": None,
            "text": s["text"],
        }
        for i, s in enumerate(raw_segments)
    ]
    write_json_atomic(out_path, {
        "doc_id": doc_id,
        "lecture": lecture_dir.name,
        "source_type": "video",
        "source_file": str(video),
        "source_sha256": source_hash,
        "model": MODEL_SIZE,
        "language": language,
        "duration": duration,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "segments": segments,
    })
    print(f"  {len(segments)} segments in {elapsed:.0f}s "
          f"(RTF {elapsed / duration:.3f}) -> {out_path}")


def main() -> None:
    args = sys.argv[1:]                    # [course] [lecture ...]
    course = args[0] if args else None
    for course_dir in course_dirs(course):
        language = load_course(course_dir)["language"]
        for lecture_dir in lecture_dirs(course_dir, args[1:]):
            process_lecture(course_dir, lecture_dir, language)



if __name__ == "__main__":
    main()