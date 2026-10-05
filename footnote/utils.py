"""Small helpers shared by the ingestion scripts."""
import hashlib
import json
from pathlib import Path


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