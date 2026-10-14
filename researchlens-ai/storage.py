"""Local persistence of document records and chat history (SQLite).

Separate from the vector store: SQLite holds human-facing bookkeeping (which
papers are indexed, when, how many pages/chunks, and the Q&A history) while the
vector store holds the embeddings. Keeping them apart makes each easy to reason
about and test.

The ``doc_id`` is a content hash (see ``pdf_processor.compute_doc_id``) so the
same PDF uploaded twice maps to the same record — this is how we avoid
re-indexing identical documents.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    num_pages   INTEGER NOT NULL,
    num_chunks  INTEGER NOT NULL,
    added_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    sources     TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class Storage:
    """Thin wrapper around a SQLite database file."""

    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = Path(db_path)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # --- documents ---------------------------------------------------------
    def document_exists(self, doc_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return row is not None

    def add_document(
        self, doc_id: str, filename: str, num_pages: int, num_chunks: int
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO documents "
                "(doc_id, filename, num_pages, num_chunks) VALUES (?, ?, ?, ?)",
                (doc_id, filename, num_pages, num_chunks),
            )

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT doc_id, filename, num_pages, num_chunks, added_at "
                "FROM documents ORDER BY added_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_document(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT doc_id, filename, num_pages, num_chunks, added_at "
                "FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            return dict(row) if row else None

    def delete_document(self, doc_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    # --- history -----------------------------------------------------------
    def add_history(self, question: str, answer: str, sources: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO history (question, answer, sources) "
                "VALUES (?, ?, ?)",
                (question, answer, sources),
            )

    def list_history(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, question, answer, sources, created_at "
                "FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_history(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history")
