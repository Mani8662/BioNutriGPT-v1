"""Extract readable text from small user-uploaded nutrition documents."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_EXTRACTED_CHARS = 12_000
MAX_PDF_PAGES = 6


class DocumentExtractionError(ValueError):
    pass


def extract_document_text(filename: str, content: bytes) -> tuple[str, bool]:
    """Return extracted text and whether the text was truncated for model context."""
    if not content:
        raise DocumentExtractionError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentExtractionError("Files must be 8 MB or smaller.")

    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".csv"}:
        text = content.decode("utf-8-sig", errors="replace")
    elif suffix in {".png", ".jpg", ".jpeg"}:
        expected = {".png": b"\x89PNG\r\n\x1a\n", ".jpg": b"\xff\xd8\xff", ".jpeg": b"\xff\xd8\xff"}[suffix]
        if not content.startswith(expected):
            raise DocumentExtractionError("The image file does not match its extension.")
        with tempfile.TemporaryDirectory(prefix="bionutrigpt-upload-") as temp_dir:
            image_path = Path(temp_dir) / f"upload{suffix}"
            image_path.write_bytes(content)
            text = _ocr_image(image_path)
    elif suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise DocumentExtractionError("The PDF file could not be recognized.")
        with tempfile.TemporaryDirectory(prefix="bionutrigpt-upload-") as temp_dir:
            pdf_path = Path(temp_dir) / "upload.pdf"
            pdf_path.write_bytes(content)
            text, pages_truncated = _ocr_pdf(pdf_path, Path(temp_dir) / "page")
    else:
        raise DocumentExtractionError("Choose a .txt, .csv, .pdf, .png, .jpg, or .jpeg file.")

    text = text.strip()
    if not text:
        raise DocumentExtractionError(
            "No readable text was found. Try a clearer image or a text-based PDF."
        )
    truncated = len(text) > MAX_EXTRACTED_CHARS
    return text[:MAX_EXTRACTED_CHARS], truncated or (suffix == ".pdf" and pages_truncated)


def _ocr_image(image_path: Path) -> str:
    try:
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", "eng"],
            check=False, capture_output=True, text=True, timeout=35,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise DocumentExtractionError("Image text reading is unavailable or timed out.") from exc
    if result.returncode:
        raise DocumentExtractionError("Could not read text from this image.")
    return result.stdout


def _ocr_pdf(pdf_path: Path, output_prefix: Path) -> tuple[str, bool]:
    try:
        subprocess.run(
            ["pdftoppm", "-f", "1", "-l", str(MAX_PDF_PAGES + 1), "-r", "180", "-png", str(pdf_path), str(output_prefix)],
            check=True, capture_output=True, text=True, timeout=45,
        )
    except FileNotFoundError as exc:
        raise DocumentExtractionError("PDF text reading requires Poppler (pdftoppm).") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise DocumentExtractionError("Could not convert this PDF into readable pages.") from exc

    pages = sorted(pdf_path.parent.glob("page-*.png"))
    if not pages:
        raise DocumentExtractionError("No readable pages were found in this PDF.")
    pages_truncated = len(pages) > MAX_PDF_PAGES
    text = "\n\n".join(_ocr_image(page) for page in pages[:MAX_PDF_PAGES])
    return text, pages_truncated
