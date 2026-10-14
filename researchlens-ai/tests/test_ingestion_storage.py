"""Tests for the ingestion orchestrator (dedup, delete) and SQLite storage."""

from __future__ import annotations

from embeddings import get_embedder
from ingestion import delete_document, ingest_pdf
from storage import Storage
from vector_store import VectorStore

# A tiny valid-looking PDF built with PyMuPDF when available; otherwise these
# ingestion tests are skipped (they need real extraction).
import pytest

fitz = pytest.importorskip("fitz")


def _make_pdf(text_by_page):
    doc = fitz.open()
    for text in text_by_page:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _fresh(tmp_path):
    storage = Storage(db_path=tmp_path / "test.db")
    store = VectorStore(dimension=get_embedder().dimension)
    return store, storage


def test_ingest_then_dedup(tmp_path):
    store, storage = _fresh(tmp_path)
    data = _make_pdf(["federated learning privacy", "second page content"])

    r1 = ingest_pdf(data, "paper.pdf", store, storage, persist=False)
    assert r1.ok and not r1.skipped
    assert r1.num_pages == 2 and r1.num_chunks >= 1
    assert storage.document_exists(r1.doc_id)

    # Re-ingesting identical bytes should be skipped.
    r2 = ingest_pdf(data, "paper.pdf", store, storage, persist=False)
    assert r2.skipped is True
    # Size unchanged (no duplicate vectors added).
    assert store.size == r1.num_chunks


def test_delete_removes_from_store_and_db(tmp_path):
    store, storage = _fresh(tmp_path)
    data = _make_pdf(["alpha beta", "gamma delta"])
    r = ingest_pdf(data, "p.pdf", store, storage, persist=False)
    assert store.size > 0

    removed = delete_document(r.doc_id, store, storage, persist=False)
    assert removed == r.num_chunks
    assert store.size == 0
    assert not storage.document_exists(r.doc_id)


def test_corrupt_pdf_reports_error(tmp_path):
    store, storage = _fresh(tmp_path)
    r = ingest_pdf(b"not a real pdf", "bad.pdf", store, storage, persist=False)
    assert not r.ok
    assert r.error


def test_storage_history_roundtrip(tmp_path):
    storage = Storage(db_path=tmp_path / "h.db")
    storage.add_history("Q1?", "A1", "paper.pdf p.1")
    storage.add_history("Q2?", "A2", "paper.pdf p.2")
    hist = storage.list_history()
    assert len(hist) == 2
    assert hist[0]["question"] == "Q2?"  # newest first
    storage.clear_history()
    assert storage.list_history() == []


def test_storage_document_listing(tmp_path):
    storage = Storage(db_path=tmp_path / "d.db")
    storage.add_document("id1", "a.pdf", 3, 7)
    docs = storage.list_documents()
    assert len(docs) == 1
    assert docs[0]["filename"] == "a.pdf"
    assert docs[0]["num_chunks"] == 7
