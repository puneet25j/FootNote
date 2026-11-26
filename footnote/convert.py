"""Turn a lecture document into a PDF, so every later step only has to read PDFs.

    .pdf                   used as it is
    .pptx .docx            LibreOffice, headless
    .png .jpg .jpeg .webp  Pillow, one page

Usage (from the Footnote folder):
    python -m footnote.convert <document> <output folder>
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

from footnote.utils import already_done, sha256_file, write_json_atomic

CONVERTER_VERSION = 1   # bump to convert every document again
OFFICE_MARKERS = {".pptx": "ppt/presentation.xml", ".docx": "word/document.xml"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SOFFICE_CANDIDATES = ["soffice", "libreoffice",
                      "/Applications/LibreOffice.app/Contents/MacOS/soffice"]


class ConvertError(RuntimeError):
    """A document could not be turned into a PDF."""


def find_soffice() -> str:
    """LibreOffice's command: $SOFFICE if set, else the usual names, else the Mac app."""
    candidates = [os.environ["SOFFICE"]] if os.environ.get("SOFFICE") else []
    for candidate in candidates + SOFFICE_CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    raise ConvertError("LibreOffice not found: install it, or set SOFFICE=/path/to/soffice")


def check_office_file(src: Path) -> None:
    """A real .pptx/.docx is a zip with a known part inside; LibreOffice would accept anything."""
    marker = OFFICE_MARKERS[src.suffix.lower()]
    try:
        with zipfile.ZipFile(src) as z:
            if marker in z.namelist():
                return
    except zipfile.BadZipFile:
        pass
    raise ConvertError(f"{src.name} is not a real {src.suffix.lower()} file")


def office_to_pdf(src: Path, out_pdf: Path) -> None:
    check_office_file(src)
    profile = Path(tempfile.gettempdir()) / "footnote-libreoffice"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = subprocess.run(
                [find_soffice(), f"-env:UserInstallation={profile.as_uri()}",
                 "--headless", "--convert-to", "pdf", "--outdir", tmp, str(src)],
                capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            raise ConvertError(f"LibreOffice took over 15 min on {src.name}") from None
        made = Path(tmp) / f"{src.stem}.pdf"
        if not made.exists():   # LibreOffice can fail and still exit with 0
            lines = (result.stdout + result.stderr).strip().splitlines()
            reason = lines[-1] if lines else f"exit code {result.returncode}"
            raise ConvertError(f"LibreOffice could not convert {src.name}: {reason}")
        shutil.move(made, out_pdf)


def image_to_pdf(src: Path, out_pdf: Path) -> None:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)      # phone photos store "rotate me" in EXIF
            white = Image.new("RGBA", im.size, "white")
            page = Image.alpha_composite(white, im.convert("RGBA")).convert("RGB")
            page.save(out_pdf, "PDF", resolution=150)
    except OSError as e:                          # not an image, or a damaged one
        raise ConvertError(f"could not read image {src.name}: {e}") from None


def to_pdf(src: Path, out_dir: Path) -> Path:
    """The PDF to read for src: src itself if it is a PDF, else a cached conversion in out_dir."""
    ext = src.suffix.lower()
    if ext == ".pdf":
        return src
    if ext not in OFFICE_MARKERS and ext not in IMAGE_EXTENSIONS:
        raise ConvertError(f"don't know how to convert {src.name}")
    out_pdf, record = out_dir / "doc.pdf", out_dir / "convert.json"
    source_hash = sha256_file(src)
    if out_pdf.exists() and already_done(record, source_hash, converter=CONVERTER_VERSION):
        print(f"skip {src.name}: already converted")
        return out_pdf

    print(f"converting {src.name} ...")
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / "doc.pdf.part"
    try:
        if ext in OFFICE_MARKERS:
            office_to_pdf(src, part)
        else:
            image_to_pdf(src, part)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(out_pdf)
    write_json_atomic(record, {"source_file": src.name, "source_sha256": source_hash,
                               "converter": CONVERTER_VERSION})   # the "done" marker goes last
    print(f"  done in {time.perf_counter() - t0:.1f}s -> {out_pdf}")
    return out_pdf


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit("usage: python -m footnote.convert <document> <output folder>")
    try:
        to_pdf(Path(sys.argv[1]), Path(sys.argv[2]))
    except ConvertError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()