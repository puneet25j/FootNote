"""Small helpers shared by the ingestion scripts."""
import hashlib
import json
from pathlib import Path

COURSES_DIR = Path("data/courses")

def sha256_file(path: Path) -> str:
    """Fingerprint a file's bytes: same bytes -> same hash."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, data: dict) -> None:
    """Write to a temp file, then swap it in, so the result is never half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def already_done(out_path: Path, source_hash: str, **settings) -> bool:
    """True if out_path was built from these exact bytes with these settings."""
    if not out_path.exists():
        return False
    old = json.loads(out_path.read_text())
    if old.get("source_sha256") != source_hash:
        return False
    return all(old.get(key) == value for key, value in settings.items())


def course_dirs(only: str | None=None) -> list[Path]:
    """Every course folder, or just the one named."""
    if only:
        return [COURSES_DIR / only]
    if not COURSES_DIR.exists():
        return []
    return sorted(p for p in COURSES_DIR.iterdir() if p.is_dir())

def lecture_dirs(course_dir: Path, only: list[str] | None = None) -> list[Path]:
    """Every lecture folder in a course's raw/ folder, or just the ones named."""
    raw = course_dir / "raw"
    if only:
        return [raw / name for name in only]
    if not raw.exists():  # e.g. a fresh clone: processed JSON but no media
        return []
    return sorted(p for p in raw.iterdir() if p.is_dir())

def load_course(course_dir: Path) -> dict:
    """A course's settings from course.json, with defaults for anything missing."""
    path = course_dir / "course.json"
    info = json.loads(path.read_text()) if path.exists() else {}
    return {
        "id": course_dir.name,
        "name": info.get("name", course_dir.name),
        "language": info.get("language", "en"),
    }