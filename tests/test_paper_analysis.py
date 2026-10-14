"""Tests for summaries, comparison, and gap analysis (LLM mocked)."""

from __future__ import annotations

from conftest import FakeLLM

from chunking import chunk_pages
from embeddings import get_embedder
from paper_analysis import (compare_papers, research_gap_analysis,
                            summarize_paper)
from vector_store import VectorStore


def _multi_doc_store():
    emb = get_embedder()
    store = VectorStore(dimension=emb.dimension)
    a = chunk_pages([(1, "federated learning privacy"), (2, "fedavg method")],
                    doc_id="docA", filename="a.pdf", chunk_size=400, overlap=0)
    b = chunk_pages([(1, "secret image sharing"), (2, "digit decomposition")],
                    doc_id="docB", filename="b.pdf", chunk_size=400, overlap=0)
    store.add(emb.encode([c.text for c in a]), a)
    store.add(emb.encode([c.text for c in b]), b)
    return store


def test_summary_runs_and_titles_by_filename(built_store):
    res = summarize_paper("docA", built_store, llm=FakeLLM(reply="## x\nSTATED"))
    assert "paperA.pdf" in res.title
    assert res.llm_used is True
    assert res.body


def test_summary_without_llm_is_handled(built_store):
    from llm_providers import NullLLM
    res = summarize_paper("docA", built_store, llm=NullLLM())
    assert res.llm_used is False


def test_compare_requires_two_papers():
    store = _multi_doc_store()
    res = compare_papers(["docA"], store, llm=FakeLLM())
    assert res.llm_used is False  # guarded, no call made


def test_compare_two_papers_produces_body():
    store = _multi_doc_store()
    res = compare_papers(["docA", "docB"], store,
                         llm=FakeLLM(reply="| Dimension | a | b |"))
    assert res.llm_used is True
    assert "Dimension" in res.body


def test_gap_analysis_prepends_disclaimer():
    store = _multi_doc_store()
    res = research_gap_analysis(["docA", "docB"], store,
                                llm=FakeLLM(reply="Some analysis without note"))
    assert res.body.strip().startswith(">")
    assert "POTENTIAL" in res.body
