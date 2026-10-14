"""Tests for the vector store: retrieval, source mapping, deletion,
document filtering, empty behaviour, and persistence round-trips."""

from __future__ import annotations

from chunking import chunk_pages
from embeddings import get_embedder
from vector_store import VectorStore


def _index(pages, doc_id, filename):
    emb = get_embedder()
    store = VectorStore(dimension=emb.dimension)
    chunks = chunk_pages(pages, doc_id=doc_id, filename=filename,
                         chunk_size=400, overlap=80)
    store.add(emb.encode([c.text for c in chunks]), chunks)
    return store, chunks


def test_search_returns_hits_with_source_mapping(built_store):
    emb = get_embedder()
    q = emb.encode(["privacy when training across devices"])[0]
    hits = built_store.search(q, top_k=3)
    assert hits, "expected at least one hit"
    top = hits[0]
    # Source mapping: hit carries filename + page from the original chunk.
    assert top.chunk.filename == "paperA.pdf"
    assert top.chunk.page in {1, 2, 3}
    # Scores are sorted descending.
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_store_returns_no_hits():
    emb = get_embedder()
    store = VectorStore(dimension=emb.dimension)
    q = emb.encode(["anything"])[0]
    assert store.search(q, top_k=5) == []


def test_document_filtering_restricts_results():
    pages_a = [(1, "federated learning privacy devices model aggregation")]
    pages_b = [(1, "secret image sharing shadow digit decomposition")]
    store, _ = _index(pages_a, "docA", "a.pdf")
    emb = get_embedder()
    chunks_b = chunk_pages(pages_b, doc_id="docB", filename="b.pdf",
                           chunk_size=400, overlap=80)
    store.add(emb.encode([c.text for c in chunks_b]), chunks_b)

    q = emb.encode(["privacy"])[0]
    hits = store.search(q, top_k=5, allowed_doc_ids={"docB"})
    assert all(h.chunk.doc_id == "docB" for h in hits)


def test_delete_document_removes_from_future_results():
    store, _ = _index(
        [(1, "federated learning privacy"), (2, "image sharing shadows")],
        "docA", "a.pdf")
    before = store.size
    removed = store.delete_document("docA")
    assert removed == before
    assert store.size == 0
    assert "docA" not in store.document_ids()


def test_persistence_round_trip(tmp_path):
    store, _ = _index([(1, "federated learning privacy devices")],
                      "docA", "a.pdf")
    idx = tmp_path / "faiss.index"
    meta = tmp_path / "metadata.json"
    store.save(idx, meta)

    loaded = VectorStore.load(idx, meta)
    assert loaded is not None
    assert loaded.size == store.size
    assert loaded.document_ids() == store.document_ids()

    emb = get_embedder()
    q = emb.encode(["privacy"])[0]
    assert loaded.search(q, top_k=1)  # still searchable after reload


def test_load_missing_returns_none(tmp_path):
    assert VectorStore.load(tmp_path / "nope.index",
                            tmp_path / "nope.json") is None


def test_get_chunks_in_reading_order(built_store):
    chunks = built_store.get_chunks("docA")
    idxs = [c.chunk_index for c in chunks]
    assert idxs == sorted(idxs)
