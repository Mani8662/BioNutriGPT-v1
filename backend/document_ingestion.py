"""Extract readable text from user-uploaded nutrition documents, lab reports, PDFs, and images."""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LOGGER = logging.getLogger("BioNutriGPT.DocumentIngestion")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_CHARS = 16_000
MAX_PDF_PAGES = 10


class DocumentExtractionError(ValueError):
    pass


def _find_tesseract_binary() -> str:
    found = shutil.which("tesseract")
    if found:
        return found
    common_paths = [
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
        "/usr/bin/tesseract",
    ]
    for p in common_paths:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return "tesseract"


def extract_document_text(filename: str, content: bytes) -> tuple[str, bool]:
    """Return extracted text and whether the text was truncated for model context."""
    if not content:
        raise DocumentExtractionError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentExtractionError("Files must be 10 MB or smaller.")

    suffix = Path(filename).suffix.lower()
    pages_truncated = False

    if suffix in {".txt", ".csv"}:
        text = content.decode("utf-8-sig", errors="replace")
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        with tempfile.TemporaryDirectory(prefix="bionutrigpt-upload-") as temp_dir:
            image_path = Path(temp_dir) / f"upload{suffix}"
            image_path.write_bytes(content)
            text = _ocr_image(image_path)
    elif suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise DocumentExtractionError("The PDF file could not be recognized.")

        # 1. Fast extraction with pypdf (pure Python, handles digital PDFs)
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(content))
            total_pages = len(reader.pages)
            pages_truncated = total_pages > MAX_PDF_PAGES
            extracted_pages = []
            for page in reader.pages[:MAX_PDF_PAGES]:
                pt = page.extract_text() or ""
                if pt.strip():
                    extracted_pages.append(pt.strip())
            text = "\n\n".join(extracted_pages).strip()
        except Exception as exc:
            LOGGER.warning("pypdf extraction failed: %s", exc)
            text = ""

        # 2. If pypdf returned no text (e.g. scanned image PDF), try OCR fallback
        if not text:
            with tempfile.TemporaryDirectory(prefix="bionutrigpt-upload-") as temp_dir:
                pdf_path = Path(temp_dir) / "upload.pdf"
                pdf_path.write_bytes(content)
                text, pages_truncated = _extract_pdf_text_or_ocr(pdf_path, Path(temp_dir) / "page")
    else:
        raise DocumentExtractionError("Choose a .txt, .csv, .pdf, .png, .jpg, .jpeg, or .webp file.")

    text = text.strip()
    if not text:
        raise DocumentExtractionError(
            "No readable text was found in this file. Try a clearer image or a text-based PDF."
        )
    truncated = len(text) > MAX_EXTRACTED_CHARS
    return text[:MAX_EXTRACTED_CHARS], truncated or (suffix == ".pdf" and pages_truncated)


def _ocr_image(image_path: Path) -> str:
    tesseract_bin = _find_tesseract_binary()
    try:
        result = subprocess.run(
            [tesseract_bin, str(image_path), "stdout", "-l", "eng", "--psm", "6"],
            check=False, capture_output=True, text=True, timeout=40,
        )
        if result.returncode != 0 or not result.stdout.strip():
            result = subprocess.run(
                [tesseract_bin, str(image_path), "stdout", "-l", "eng"],
                check=False, capture_output=True, text=True, timeout=40,
            )
    except FileNotFoundError as exc:
        raise DocumentExtractionError("Tesseract OCR is not installed or not found on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise DocumentExtractionError("Image text reading timed out.") from exc

    if result.returncode != 0 and not result.stdout.strip():
        raise DocumentExtractionError(f"Could not read text from this image: {result.stderr.strip() or 'OCR error'}")
    return result.stdout


def _extract_pdf_text_or_ocr(pdf_path: Path, output_prefix: Path) -> tuple[str, bool]:
    """Use fast embedded-text extraction for digital PDFs, OCR for scans."""
    try:
        info = subprocess.run(
            ["pdfinfo", str(pdf_path)], check=False, capture_output=True, text=True, timeout=8,
        )
        page_match = re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)
        page_count = int(page_match.group(1)) if info.returncode == 0 and page_match else None
        extracted = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(MAX_PDF_PAGES), "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
            check=False, capture_output=True, text=True, timeout=12,
        )
        if extracted.returncode == 0 and extracted.stdout.strip():
            return extracted.stdout, bool(page_count and page_count > MAX_PDF_PAGES)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return _ocr_pdf(pdf_path, output_prefix)


def _ocr_pdf(pdf_path: Path, output_prefix: Path) -> tuple[str, bool]:
    try:
        subprocess.run(
            ["pdftoppm", "-f", "1", "-l", str(MAX_PDF_PAGES + 1), "-r", "180", "-png", str(pdf_path), str(output_prefix)],
            check=True, capture_output=True, text=True, timeout=45,
        )
    except FileNotFoundError as exc:
        raise DocumentExtractionError("Scanned PDF reading requires Poppler (pdftoppm) or a text-based PDF.") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise DocumentExtractionError("Could not convert this PDF into readable pages.") from exc

    pages = sorted(pdf_path.parent.glob("page-*.png"))
    if not pages:
        raise DocumentExtractionError("No readable pages were found in this PDF.")
    pages_truncated = len(pages) > MAX_PDF_PAGES
    selected_pages = pages[:MAX_PDF_PAGES]
    with ThreadPoolExecutor(max_workers=min(4, len(selected_pages))) as executor:
        page_text = list(executor.map(_ocr_image, selected_pages))
    text = "\n\n".join(page_text)
    return text, pages_truncated
