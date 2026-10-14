"""PDF text extraction for ResearchLens AI (PyMuPDF / fitz).

Extracts text page by page, preserving the page number, and computes a stable
content hash used as the document id (so identical uploads are detected).

PyMuPDF is imported lazily inside the functions that need it, so this module
imports cleanly even where PyMuPDF isn't installed — the pure helpers
(hashing, validation) remain importable and testable.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PDFProcessingError(Exception):
    """Raised for corrupt, empty, unsupported, or unreadable PDFs."""


@dataclass
class ExtractedPage:
    page: int   # 1-based
    text: str


def compute_doc_id(data: bytes) -> str:
    """Stable SHA-256 content hash of the raw PDF bytes.

    Same bytes -> same id, so the app can skip re-indexing a document it has
    already processed.
    """
    return hashlib.sha256(data).hexdigest()[:16]


def _validate(data: bytes, filename: str) -> None:
    if not data:
        raise PDFProcessingError(f"'{filename}' is empty.")
    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError(
            f"'{filename}' is not a .pdf file (unsupported type)."
        )
    # A real PDF starts with the magic bytes '%PDF'.
    if not data[:5].startswith(b"%PDF"):
        raise PDFProcessingError(
            f"'{filename}' does not look like a valid PDF (bad header)."
        )


def extract_pages(data: bytes, filename: str) -> list[ExtractedPage]:
    """Extract text from each page of a PDF given as raw bytes.

    Raises :class:`PDFProcessingError` on any problem. Pages that contain no
    extractable text (e.g. pure images) are returned with empty text rather
    than dropped, so page numbering stays faithful to the original.
    """
    _validate(data, filename)

    try:
        import fitz  # PyMuPDF, lazy import
    except ImportError as exc:  # pragma: no cover - env-specific
        raise PDFProcessingError(
            "PyMuPDF is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise PDFProcessingError(
            f"Could not open '{filename}': {exc}"
        ) from exc

    try:
        if doc.page_count == 0:
            raise PDFProcessingError(f"'{filename}' has no pages.")
        pages: list[ExtractedPage] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = page.get_text("text") or ""
            pages.append(ExtractedPage(page=i + 1, text=text.strip()))
    finally:
        doc.close()

    if all(not p.text for p in pages):
        raise PDFProcessingError(
            f"No extractable text found in '{filename}'. It may be a scanned "
            f"document that needs OCR."
        )
    return pages


def pages_as_tuples(pages: list[ExtractedPage]) -> list[tuple[int, str]]:
    """Adapt :class:`ExtractedPage` list to the ``(page, text)`` tuples that
    :func:`chunking.chunk_pages` expects."""
    return [(p.page, p.text) for p in pages]
