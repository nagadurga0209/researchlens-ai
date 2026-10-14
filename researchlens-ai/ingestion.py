"""Ingestion orchestrator: PDF bytes -> indexed, searchable document.

Ties the pieces together so the UI (and tests) call one function:
    extract (pdf_processor) -> chunk (chunking) -> embed (embeddings)
    -> add to VectorStore -> record in Storage.

Handles deduplication (identical bytes -> same doc_id -> skip re-index) and
keeps the SQLite records and the vector store consistent, including on delete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import config
from chunking import chunk_pages
from embeddings import get_embedder
from pdf_processor import PDFProcessingError, compute_doc_id, extract_pages, \
    pages_as_tuples
from storage import Storage
from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    skipped: bool          # True if it was already indexed (dedup)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def load_or_create_store(storage: Storage) -> VectorStore:
    """Load a persisted store if present, otherwise create an empty one sized
    to the current embedder."""
    store = VectorStore.load()
    if store is None:
        store = VectorStore(dimension=get_embedder().dimension)
    return store


def ingest_pdf(
    data: bytes,
    filename: str,
    store: VectorStore,
    storage: Storage,
    *,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    persist: bool = True,
) -> IngestResult:
    """Index a single PDF. Safe to call repeatedly: duplicates are skipped."""
    doc_id = compute_doc_id(data)

    if storage.document_exists(doc_id) and doc_id in store.document_ids():
        rec = storage.get_document(doc_id) or {}
        logger.info("Skipping already-indexed document %s", filename)
        return IngestResult(
            doc_id=doc_id,
            filename=rec.get("filename", filename),
            num_pages=rec.get("num_pages", 0),
            num_chunks=rec.get("num_chunks", 0),
            skipped=True,
        )

    try:
        pages = extract_pages(data, filename)
    except PDFProcessingError as exc:
        logger.warning("Failed to ingest %s: %s", filename, exc)
        return IngestResult(doc_id, filename, 0, 0, skipped=False,
                            error=str(exc))

    chunks = chunk_pages(
        pages_as_tuples(pages),
        doc_id=doc_id,
        filename=filename,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    if not chunks:
        return IngestResult(doc_id, filename, len(pages), 0, skipped=False,
                            error="No text chunks were produced.")

    vectors = get_embedder().encode([c.text for c in chunks])
    store.add(vectors, chunks)
    storage.add_document(doc_id, filename, len(pages), len(chunks))

    if persist:
        store.save()

    logger.info("Indexed %s: %d pages, %d chunks", filename, len(pages),
                len(chunks))
    return IngestResult(doc_id, filename, len(pages), len(chunks),
                        skipped=False)


def delete_document(
    doc_id: str, store: VectorStore, storage: Storage, *, persist: bool = True
) -> int:
    """Remove a document from BOTH the vector store and the database."""
    removed = store.delete_document(doc_id)
    storage.delete_document(doc_id)
    if persist:
        store.save()
    logger.info("Deleted document %s (%d chunks removed)", doc_id, removed)
    return removed
