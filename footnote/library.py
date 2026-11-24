"""Create and manage courses and lectures on disk.

The UI calls these functions; nothing here imports Streamlit, so a later API can reuse them.
Ids (folder names) are internal; names and titles are what people see.
"""
import json
import re
import shutil
import unicodedata
from pathlib import Path
from typing import BinaryIO

from footnote.utils import COURSES_DIR, MEDIA_EXTENSIONS,course_dirs, write_json_atomic

DOC_EXTENSIONS = {".pdf", ".pptx", ".docx", ".png", ".jpg", ".jpeg", ".webp"}

def _clean(text: str, what: str) -> str:
    """Strip text; refuse it if nothing is left."""
    text = text.strip()
    if not text:
        raise ValueError(f"{what} cannot be empty")
    return text


def slugify(name: str) -> str:
    """'Álgebra Lineal (2026)' -> 'algebra-lineal-2026'. Safe as a folder name."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")
    return slug or "course"


def create_course(name: str, language: str = "en") -> str:
    """Make an empty course folder with its course.json; return the new course id."""
    name = _clean(name, "Course name")
    base = slugify(name)
    course_id, n = base, 2
    while (COURSES_DIR / course_id).exists():
        course_id, n = f"{base}-{n}", n + 1
    course_dir = COURSES_DIR / course_id
    (course_dir / "raw").mkdir(parents=True)
    write_json_atomic(course_dir / "course.json",
                      {"name": name, "language": language, "lectures": {}})
    return course_id

def _course_json(course_id: str) -> Path:
    path = COURSES_DIR / course_id / "course.json"
    if not path.exists():
        raise ValueError(f"no course with id {course_id!r}")
    return path


def _read(course_id: str) -> dict:
    """course.json exactly as stored (not load_course, which adds 'id' and defaults)."""
    return json.loads(_course_json(course_id).read_text())


def _write(course_id: str, info: dict) -> None:
    write_json_atomic(_course_json(course_id), info)

def _lecture(info: dict, course_id: str, lecture_id: str) -> dict:
    """The lecture's entry in course.json; raises if it isn't there."""
    lecture = info["lectures"].get(lecture_id)
    if lecture is None:
        raise ValueError(f"no lecture {lecture_id!r} in course {course_id!r}")
    return lecture


def _next_id(prefix: str, existing: set[str]) -> str:
    """Next id after the highest existing one: ('lec', {'lec02', 'lec03'}) -> 'lec04'."""
    highest = 0
    for item in existing:
        m = re.fullmatch(rf"{prefix}(\d+)", item)
        if m:
            highest = max(highest, int(m.group(1)))
    return f"{prefix}{highest + 1:02d}"


def add_lecture(course_id: str, title: str, video_url: str | None = None) -> str:
    """Add an empty lecture to a course; return its id (lec01, lec02, ...)."""
    title = _clean(title, "Lecture title")
    info = _read(course_id)
    lectures = info.setdefault("lectures", {})
    raw = COURSES_DIR / course_id / "raw"
    existing = set(lectures) | {p.name for p in raw.iterdir() if p.is_dir()}
    lecture_id = _next_id("lec", existing)
    (raw / lecture_id).mkdir()
    lectures[lecture_id] = {"title": title, "video": video_url, "docs": {}}
    _write(course_id, info)
    return lecture_id

def _lecture_dir(course_id: str, lecture_id: str) -> Path:
    lecture_dir = COURSES_DIR / course_id / "raw" / lecture_id
    if not lecture_dir.is_dir():
        raise ValueError(f"no lecture {lecture_id!r} in course {course_id!r}")
    return lecture_dir


def _write_part(fileobj: BinaryIO, dest: Path) -> Path:
    """Copy an upload to dest.part in 1 MB pieces; return the .part path."""
    part = dest.with_name(dest.name + ".part")
    try:
        with part.open("wb") as out:
            shutil.copyfileobj(fileobj, out, length=1024 * 1024)
    except BaseException:
        part.unlink(missing_ok=True)   # don't leave half an upload behind
        raise
    return part


def save_recording(course_id: str, lecture_id: str, filename: str, fileobj: BinaryIO) -> Path:
    """Store the lecture's recording as recording<ext>, replacing any previous one."""
    ext = Path(filename).suffix.lower()
    if ext not in MEDIA_EXTENSIONS:
        raise ValueError(f"{filename!r} is not a supported recording "
                         f"({', '.join(sorted(MEDIA_EXTENSIONS))})")
    lecture_dir = _lecture_dir(course_id, lecture_id)
    dest = lecture_dir / f"recording{ext}"
    part = _write_part(fileobj, dest)        # 1. new recording, invisible as .part
    for old in lecture_dir.iterdir():        # 2. remove the previous recording
        if old.suffix.lower() in MEDIA_EXTENSIONS and old != dest:
            old.unlink()
    part.replace(dest)                       # 3. swap the new one in
    return dest


def add_document(course_id: str, lecture_id: str, title: str,
                 filename: str, fileobj: BinaryIO) -> str:
    """Store an uploaded document as docs/dNN<ext>; return its id (d01, d02, ...)."""
    ext = Path(filename).suffix.lower()
    if ext not in DOC_EXTENSIONS:
        raise ValueError(f"{filename!r}: documents must be "
                         f"{', '.join(sorted(DOC_EXTENSIONS))}")
    lecture_dir = _lecture_dir(course_id, lecture_id)
    info = _read(course_id)
    lecture = _lecture(info, course_id, lecture_id)
    docs = lecture.setdefault("docs", {})
    docs_dir = lecture_dir / "docs"
    docs_dir.mkdir(exist_ok=True)
    doc_id = _next_id("d", set(docs) | {p.stem for p in docs_dir.iterdir()})
    dest = docs_dir / f"{doc_id}{ext}"
    _write_part(fileobj, dest).replace(dest)
    docs[doc_id] = {"title": title.strip() or Path(filename).stem, "file": dest.name}
    _write(course_id, info)
    return doc_id

def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _docs(info: dict, course_id: str, lecture_id: str, doc_id: str) -> dict:
    """The lecture's docs dict; raises unless doc_id is in it."""
    docs = _lecture(info, course_id, lecture_id).get("docs", {})
    if doc_id not in docs:
        raise ValueError(f"no document {doc_id!r} in {course_id}/{lecture_id}")
    return docs


def rename_course(course_id: str, name: str) -> None:
    info = _read(course_id)
    info["name"] = _clean(name, "Course name")
    _write(course_id, info)


def rename_lecture(course_id: str, lecture_id: str, title: str) -> None:
    info = _read(course_id)
    _lecture(info, course_id, lecture_id)["title"] = _clean(title, "Lecture title")
    _write(course_id, info)


def rename_document(course_id: str, lecture_id: str, doc_id: str, title: str) -> None:
    info = _read(course_id)
    _docs(info, course_id, lecture_id, doc_id)[doc_id]["title"] = _clean(title, "Document title")
    _write(course_id, info)


def delete_document(course_id: str, lecture_id: str, doc_id: str) -> None:
    """Forget a document, then remove its file and anything built from it."""
    info = _read(course_id)
    doc = _docs(info, course_id, lecture_id, doc_id).pop(doc_id)
    _write(course_id, info)                                    # 1. the record first
    course_dir = COURSES_DIR / course_id
    (course_dir / "raw" / lecture_id / "docs" / doc["file"]).unlink(missing_ok=True)
    _remove_tree(course_dir / "processed" / lecture_id / "docs" / doc_id)   # 2. then files


def delete_lecture(course_id: str, lecture_id: str) -> None:
    """Forget a lecture, then remove its uploads and processed files."""
    info = _read(course_id)
    _lecture(info, course_id, lecture_id)          # raises if it isn't ours
    del info["lectures"][lecture_id]
    _write(course_id, info)
    for part in ("raw", "processed"):
        _remove_tree(COURSES_DIR / course_id / part / lecture_id)


def delete_course(course_id: str) -> None:
    """Remove a course and everything in it."""
    if course_id not in {p.name for p in course_dirs()}:
        raise ValueError(f"no course with id {course_id!r}")
    shutil.rmtree(COURSES_DIR / course_id)