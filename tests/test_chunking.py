"""Tests for text chunking and page-metadata preservation."""

from __future__ import annotations

import pytest

from chunking import chunk_pages, chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("    ") == []


def test_chunks_respect_size_and_do_not_split_words():
    text = " ".join(["word"] * 300)  # 300 words, "word" = 4 chars
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 100
        # No partial words: every token is exactly "word".
        assert all(tok == "word" for tok in c.split())


def test_overlap_shares_content_between_consecutive_chunks():
    # Distinct tokens so we can detect shared tail/head.
    text = " ".join(f"t{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=40, overlap=15)
    assert len(chunks) >= 2
    first_tail = set(chunks[0].split()[-2:])
    second_head = set(chunks[1].split()[:3])
    assert first_tail & second_head, "expected overlap between chunks"


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)


def test_chunk_pages_preserves_page_numbers_and_filename():
    pages = [(1, "alpha beta gamma"), (2, "delta epsilon zeta")]
    chunks = chunk_pages(pages, doc_id="d1", filename="f.pdf",
                         chunk_size=50, overlap=0)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.filename == "f.pdf" for c in chunks)
    assert all(c.doc_id == "d1" for c in chunks)
    # chunk_index is unique and sequential.
    idxs = [c.chunk_index for c in chunks]
    assert idxs == list(range(len(chunks)))
