"""Vector store for ResearchLens AI.

Primary backend: **FAISS** (`IndexFlatIP`). Because every vector is
L2-normalized before insertion, the inner product FAISS returns IS the cosine
similarity, in the range [-1, 1] — higher means more relevant. "Flat" means an
exact (brute-force) search, which is the right choice for the modest number of
chunks in a paper library and avoids approximate-search tuning.

Fallback backend: a NumPy matrix with the identical search math, used
automatically when `faiss` cannot be imported (e.g. offline sandbox/CI) so the
pipeline runs everywhere. The persisted files record which backend wrote them.

The store keeps a parallel list of chunk metadata, so search result row *i*
maps back to its source document, page, and text. Both the vectors and the
metadata are persisted to disk and reloaded on startup, and deleting a document
removes its vectors from future results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

import config
from chunking import Chunk

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore

    _HAS_FAISS = True
except Exception:  # noqa: BLE001
    _HAS_FAISS = False


class SearchHit:
    """One retrieval result: the chunk plus its similarity score."""

    __slots__ = ("chunk", "score")

    def __init__(self, chunk: Chunk, score: float):
        self.chunk = chunk
        self.score = float(score)

    def to_dict(self) -> dict:
        d = self.chunk.to_dict()
        d["score"] = self.score
        return d


class VectorStore:
    """An append-only, persistable similarity index over chunk embeddings."""

    def __init__(self, dimension: int):
        self.dimension = int(dimension)
        self.backend = "faiss" if _HAS_FAISS else "numpy"
        self._metas: list[Chunk] = []
        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(self.dimension)
            self._matrix = None
        else:
            self._index = None
            # lazily grown (n, dim) float32 matrix
            self._matrix = np.zeros((0, self.dimension), dtype="float32")

    # --- size helpers ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._metas)

    @property
    def size(self) -> int:
        return len(self._metas)

    # --- mutation ----------------------------------------------------------
    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Add embeddings and their matching chunk metadata."""
        if len(chunks) == 0:
            return
        vectors = np.ascontiguousarray(vectors.astype("float32"))
        if vectors.shape[0] != len(chunks):
            raise ValueError("vectors and chunks length mismatch")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected dim {self.dimension}, got {vectors.shape[1]}"
            )
        if _HAS_FAISS:
            self._index.add(vectors)
        else:
            self._matrix = np.vstack([self._matrix, vectors])
        self._metas.extend(chunks)

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to ``doc_id``.

        FAISS's flat index has no stable per-vector delete, so we rebuild the
        index from the surviving vectors — simple and correct for this scale.
        Returns the number of chunks removed.
        """
        keep_idx = [i for i, c in enumerate(self._metas) if c.doc_id != doc_id]
        removed = len(self._metas) - len(keep_idx)
        if removed == 0:
            return 0

        if _HAS_FAISS:
            surviving = (
                self._reconstruct_all()[keep_idx]
                if keep_idx
                else np.zeros((0, self.dimension), dtype="float32")
            )
            self._index = faiss.IndexFlatIP(self.dimension)
            if len(keep_idx):
                self._index.add(np.ascontiguousarray(surviving))
        else:
            self._matrix = (
                self._matrix[keep_idx]
                if keep_idx
                else np.zeros((0, self.dimension), dtype="float32")
            )
        self._metas = [self._metas[i] for i in keep_idx]
        return removed

    def _reconstruct_all(self) -> np.ndarray:
        """Return all stored vectors as a matrix (FAISS backend)."""
        n = self._index.ntotal
        if n == 0:
            return np.zeros((0, self.dimension), dtype="float32")
        return self._index.reconstruct_n(0, n)

    # --- search ------------------------------------------------------------
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = config.TOP_K,
        allowed_doc_ids: set[str] | None = None,
    ) -> list[SearchHit]:
        """Return up to ``top_k`` most similar chunks.

        ``allowed_doc_ids`` optionally restricts results to a subset of
        documents (the papers the user selected in the UI). We over-fetch and
        then filter so the final list still has up to ``top_k`` items.
        """
        if self.size == 0 or top_k <= 0:
            return []

        q = np.ascontiguousarray(query_vector.astype("float32")).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError("query dimension mismatch")

        # Over-fetch to leave room for document filtering.
        fetch = self.size if allowed_doc_ids else min(top_k, self.size)
        if allowed_doc_ids:
            fetch = min(self.size, max(top_k * 5, top_k))

        if _HAS_FAISS:
            scores, idxs = self._index.search(q, fetch)
            scores, idxs = scores[0], idxs[0]
        else:
            sims = (self._matrix @ q[0])  # cosine, since all are normalized
            idxs = np.argsort(-sims)[:fetch]
            scores = sims[idxs]

        hits: list[SearchHit] = []
        for score, i in zip(scores, idxs):
            if i < 0:
                continue
            chunk = self._metas[int(i)]
            if allowed_doc_ids and chunk.doc_id not in allowed_doc_ids:
                continue
            hits.append(SearchHit(chunk, float(score)))
            if len(hits) >= top_k:
                break
        return hits

    # --- persistence -------------------------------------------------------
    def save(
        self,
        index_path: Path = config.FAISS_INDEX_PATH,
        metadata_path: Path = config.METADATA_PATH,
    ) -> None:
        """Persist vectors + metadata so indexing survives a restart."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if _HAS_FAISS:
            faiss.write_index(self._index, str(index_path))
        else:
            # Store the raw matrix next to where the faiss index would live.
            np.save(index_path.with_suffix(".npy"), self._matrix)

        payload = {
            "dimension": self.dimension,
            "backend": self.backend,
            "chunks": [c.to_dict() for c in self._metas],
        }
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Saved vector store (%d chunks, %s backend)",
                    self.size, self.backend)

    @classmethod
    def load(
        cls,
        index_path: Path = config.FAISS_INDEX_PATH,
        metadata_path: Path = config.METADATA_PATH,
    ) -> "VectorStore | None":
        """Load a previously saved store, or ``None`` if none exists."""
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            return None

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        store = cls(dimension=int(payload["dimension"]))
        store._metas = [Chunk.from_dict(d) for d in payload["chunks"]]

        if _HAS_FAISS and index_path.exists():
            store._index = faiss.read_index(str(index_path))
        elif _HAS_FAISS and index_path.with_suffix(".npy").exists():
            # Saved by numpy backend but we now have faiss: rebuild index.
            mat = np.load(index_path.with_suffix(".npy"))
            store._index = faiss.IndexFlatIP(store.dimension)
            if len(mat):
                store._index.add(np.ascontiguousarray(mat.astype("float32")))
        elif not _HAS_FAISS and index_path.with_suffix(".npy").exists():
            store._matrix = np.load(index_path.with_suffix(".npy"))
        elif not _HAS_FAISS and index_path.exists():
            # A faiss index on disk but no faiss here: can't read it.
            logger.warning("FAISS index present but faiss not installed; "
                           "metadata loaded without vectors.")
            store._matrix = np.zeros((0, store.dimension), dtype="float32")

        logger.info("Loaded vector store (%d chunks, %s backend)",
                    store.size, store.backend)
        return store

    def document_ids(self) -> set[str]:
        return {c.doc_id for c in self._metas}

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """All chunks for a document, in reading order (by chunk_index)."""
        chunks = [c for c in self._metas if c.doc_id == doc_id]
        return sorted(chunks, key=lambda c: c.chunk_index)

    def filename_for(self, doc_id: str) -> str:
        for c in self._metas:
            if c.doc_id == doc_id:
                return c.filename
        return doc_id
