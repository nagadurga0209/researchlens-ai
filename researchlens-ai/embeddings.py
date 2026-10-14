"""Embedding generation for ResearchLens AI.

The **same** embedder is used for document chunks and for user queries — this
is essential: query and document vectors must live in the same space for
similarity search to be meaningful.

Primary backend: Sentence-Transformers (``all-MiniLM-L6-v2`` by default),
which produces 384-dimensional vectors.

Fallback backend: a deterministic, dependency-light hashing embedder that
activates automatically when Sentence-Transformers (or its model download)
is unavailable — e.g. an offline machine or a CI sandbox. It lets the whole
pipeline run and be tested anywhere. It is NOT as good as a real model and
the app surfaces a warning when it is in use.

All embedders L2-normalize their output so that an inner-product search in
the vector store is exactly cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Protocol

import numpy as np

# Token pattern for the fallback embedder: alphanumeric runs, so punctuation
# like a trailing period in "privacy." does not block a match with "privacy".
_TOKEN_RE = re.compile(r"[a-z0-9]+")

import config

logger = logging.getLogger(__name__)

# Dimension of the default MiniLM model. The fallback matches it so indexes
# built either way are the same shape.
_DEFAULT_DIM = 384


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; zero rows are left as zeros."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


class Embedder(Protocol):
    """Minimal interface every embedding backend implements."""

    name: str
    dimension: int
    is_fallback: bool

    def encode(self, texts: list[str]) -> np.ndarray:
        ...


class SentenceTransformerEmbedder:
    """Real embeddings via the sentence-transformers library."""

    is_fallback = False

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype="float32")
        vecs = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # already unit-length
        )
        return vecs.astype("float32")


class HashingEmbedder:
    """Deterministic fallback embedder using the hashing trick.

    Each token is hashed into one of ``dimension`` buckets and contributes a
    signed count. The resulting bag-of-hashed-words vector is L2-normalized.
    It captures lexical overlap (shared words -> higher cosine) which is good
    enough to exercise and test the retrieval pipeline, but it has no semantic
    understanding. Real deployments use the model backend above.
    """

    is_fallback = True

    def __init__(self, dimension: int = _DEFAULT_DIM):
        self.name = f"hashing-fallback-{dimension}d"
        self.dimension = dimension

    def _hash(self, token: str) -> tuple[int, float]:
        h = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") % self.dimension
        sign = 1.0 if h[4] & 1 else -1.0
        return bucket, sign

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                bucket, sign = self._hash(token)
                out[i, bucket] += sign
        return _l2_normalize(out)


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Return a process-wide singleton embedder.

    Tries the real model first; on any import/load failure, falls back to the
    hashing embedder and logs a warning. Cached so the (expensive) model loads
    only once.
    """
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    try:
        _EMBEDDER = SentenceTransformerEmbedder()
        logger.info("Using SentenceTransformer embedder: %s", _EMBEDDER.name)
    except Exception as exc:  # noqa: BLE001 - we intentionally catch all
        logger.warning(
            "sentence-transformers unavailable (%s); falling back to the "
            "deterministic hashing embedder. Install requirements for real "
            "semantic search.",
            exc,
        )
        _EMBEDDER = HashingEmbedder()
    return _EMBEDDER


def reset_embedder() -> None:
    """Clear the cached embedder (used by tests)."""
    global _EMBEDDER
    _EMBEDDER = None


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convenience: embed a list of strings with the shared embedder."""
    return get_embedder().encode(texts)
