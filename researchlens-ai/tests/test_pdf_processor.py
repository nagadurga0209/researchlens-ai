"""Tests for PDF handling: validation, hashing, and (when PyMuPDF is
installed) real text extraction with page metadata."""

from __future__ import annotations

import pytest

from pdf_processor import (PDFProcessingError, compute_doc_id, extract_pages,
                           pages_as_tuples)


def test_doc_id_is_stable_and_content_addressed():
    a = compute_doc_id(b"%PDF-1.7 hello")
    b = compute_doc_id(b"%PDF-1.7 hello")
    c = compute_doc_id(b"%PDF-1.7 world")
    assert a == b          # same bytes -> same id (enables dedup)
    assert a != c          # different bytes -> different id
    assert len(a) == 16


def test_empty_file_rejected():
    with pytest.raises(PDFProcessingError):
        extract_pages(b"", "empty.pdf")


def test_non_pdf_extension_rejected():
    with pytest.raises(PDFProcessingError):
        extract_pages(b"%PDF-1.7 ...", "notes.txt")


def test_bad_header_rejected():
    with pytest.raises(PDFProcessingError):
        extract_pages(b"this is not a pdf", "fake.pdf")


def test_real_extraction_preserves_pages():
    """Full extraction round-trip. Skipped if PyMuPDF isn't installed."""
    fitz = pytest.importorskip("fitz")  # PyMuPDF

    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Hello from page one about federated learning.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Page two discusses secret image sharing.")
    data = doc.tobytes()
    doc.close()

    pages = extract_pages(data, "sample.pdf")
    assert len(pages) == 2
    assert pages[0].page == 1 and pages[1].page == 2
    assert "federated" in pages[0].text.lower()
    assert "image sharing" in pages[1].text.lower()

    tuples = pages_as_tuples(pages)
    assert tuples[0][0] == 1 and isinstance(tuples[0][1], str)
