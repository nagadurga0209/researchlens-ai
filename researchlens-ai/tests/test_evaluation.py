"""Tests for the evaluation metrics — computed from real retrieval output."""

from __future__ import annotations

from chunking import chunk_pages
from embeddings import get_embedder
from evaluation import EvalItem, evaluate_item, run_evaluation
from vector_store import VectorStore


def _eval_store():
    emb = get_embedder()
    store = VectorStore(dimension=emb.dimension)
    pages = [
        (1, "federated learning preserves privacy by keeping data on devices"),
        (2, "the method aggregates client updates with weighted averaging"),
        (3, "evaluated on mnist and cifar10 reporting accuracy"),
    ]
    chunks = chunk_pages(pages, doc_id="paper.pdf", filename="paper.pdf",
                         chunk_size=200, overlap=0)
    store.add(emb.encode([c.text for c in chunks]), chunks)
    return store


def test_perfect_retrieval_metrics():
    store = _eval_store()
    item = EvalItem("how are client updates aggregated?",
                    relevant=[("paper.pdf", 2)])
    m = evaluate_item(item, store, top_k=3, match_key="filename")
    # The relevant page should be retrieved.
    assert m.has_citation is True
    assert m.recall_at_k == 1.0
    assert 0 < m.reciprocal_rank <= 1.0
    assert m.latency_s >= 0.0


def test_irrelevant_question_scores_zero():
    store = _eval_store()
    item = EvalItem("what is the recipe for a chocolate cake?",
                    relevant=[("paper.pdf", 99)])  # page that can't match
    m = evaluate_item(item, store, top_k=3, match_key="filename")
    assert m.recall_at_k == 0.0
    assert m.reciprocal_rank == 0.0
    assert m.has_citation is False


def test_precision_bounded_by_relevant_count():
    store = _eval_store()
    item = EvalItem("privacy on devices", relevant=[("paper.pdf", 1)])
    m = evaluate_item(item, store, top_k=3, match_key="filename")
    # Only 1 relevant item among up-to-3 retrieved -> precision <= 1/1..3.
    assert 0.0 <= m.precision_at_k <= 1.0


def test_report_summary_aggregates():
    store = _eval_store()
    items = [
        EvalItem("how are client updates aggregated?", [("paper.pdf", 2)]),
        EvalItem("which datasets are used?", [("paper.pdf", 3)]),
    ]
    report = run_evaluation(items, store, top_k=3, chunk_size=200,
                            chunk_overlap=0)
    s = report.summary()
    assert s["num_queries"] == 2
    assert 0.0 <= s["MRR"] <= 1.0
    assert 0.0 <= s["citation_coverage"] <= 1.0
    assert s["chunk_size"] == 200  # records the value actually used
    df = report.to_dataframe()
    assert len(df) == 2
    assert "recall@k" in df.columns
