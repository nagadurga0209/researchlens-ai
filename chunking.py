"""Text chunking for ResearchLens AI.

Splits page-level text into overlapping, word-aware chunks. Each chunk keeps
the source filename and page number so retrieved evidence can be cited.

Why overlapping chunks? A single idea often straddles a chunk boundary.
Overlap means a sentence cut off at the end of one chunk reappears at the
start of the next, so the retriever doesn't lose it.

This module is pure Python (no heavy dependencies) and is fully unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import config


@dataclass
class Chunk:
    """A single retrievable unit of text with its provenance."""

    text: str
    doc_id: str          # stable id of the source document (content hash)
    filename: str        # original PDF filename, for display/citation
    page: int            # 1-based page number the chunk came from
    chunk_index: int     # position of this chunk within the document
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(
            text=d["text"],
            doc_id=d["doc_id"],
            filename=d["filename"],
            page=int(d["page"]),
            chunk_index=int(d["chunk_index"]),
            metadata=d.get("metadata", {}),
        )


def _normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace so chunk sizes are predictable."""
    return " ".join(text.split())


def chunk_text(
    text: str,
    *,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split a string into overlapping chunks of roughly ``chunk_size`` chars.

    Splitting is word-aware: we never cut a word in half. ``overlap`` is the
    number of characters (approx.) shared between consecutive chunks.

    Returns an empty list for empty/whitespace-only input.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        # Guard against an infinite loop / no forward progress.
        raise ValueError("overlap must be smaller than chunk_size")

    text = _normalize_whitespace(text)
    if not text:
        return []

    words = text.split(" ")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        # +1 accounts for the space that joins words.
        added_len = len(word) + (1 if current else 0)
        if current_len + added_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Build the overlap tail from the end of the current chunk.
            current, current_len = _overlap_tail(current, overlap)
            # Now add the word that didn't fit.
            current.append(word)
            current_len += len(word) + (1 if len(current) > 1 else 0)
        else:
            current.append(word)
            current_len += added_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def _overlap_tail(words: list[str], overlap: int) -> tuple[list[str], int]:
    """Return the trailing words of ``words`` whose length is ~``overlap``."""
    if overlap == 0:
        return [], 0
    tail: list[str] = []
    length = 0
    for word in reversed(words):
        added = len(word) + (1 if tail else 0)
        if length + added > overlap:
            break
        tail.insert(0, word)
        length += added
    return tail, length


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    *,
    doc_id: str,
    filename: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk an iterable of ``(page_number, page_text)`` pairs.

    Page metadata is preserved: every produced :class:`Chunk` records the
    page it came from. ``chunk_index`` is assigned sequentially across the
    whole document so each chunk has a unique, stable position.
    """
    results: list[Chunk] = []
    idx = 0
    for page_number, page_text in pages:
        for piece in chunk_text(page_text, chunk_size=chunk_size, overlap=overlap):
            results.append(
                Chunk(
                    text=piece,
                    doc_id=doc_id,
                    filename=filename,
                    page=page_number,
                    chunk_index=idx,
                )
            )
            idx += 1
    return results
