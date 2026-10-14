"""Tests for embedding generation (same model for docs and queries)."""

from __future__ import annotations

import numpy as np

from embeddings import get_embedder


def test_embeddings_have_expected_shape_and_are_normalized():
    emb = get_embedder()
    vecs = emb.encode(["hello world", "another sentence", "third one"])
    assert vecs.shape == (3, emb.dimension)
    assert vecs.dtype == np.float32
    # L2-normalized -> norms ~ 1 (zero vectors allowed for empty text only).
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_encode_empty_list_returns_empty_matrix():
    emb = get_embedder()
    vecs = emb.encode([])
    assert vecs.shape == (0, emb.dimension)


def test_same_embedder_instance_is_cached():
    assert get_embedder() is get_embedder()


def test_similar_texts_score_higher_than_dissimilar():
    """Even the fallback embedder should rank lexical overlap higher."""
    emb = get_embedder()
    v = emb.encode([
        "federated learning privacy on devices",
        "federated learning keeps data private on the device",
        "a recipe for chocolate chip cookies",
    ])
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated
