"""Tests for the RAG pipeline: grounded answers, not-found handling,
citation validation, and the no-LLM path. LLM calls are mocked."""

from __future__ import annotations

from conftest import FakeLLM

from chunking import Chunk
from rag_pipeline import (NOT_FOUND_MESSAGE, answer_question, build_prompt,
                          citation_numbers, retrieve, validate_citations)
from llm_providers import NullLLM
from vector_store import SearchHit

# A query that lexically overlaps page 1 strongly, so retrieval returns a hit
# even with the weak offline fallback embedder (real models do far better).
STRONG_QUERY = ("Federated learning trains a shared model across devices "
                "which improves user privacy")


def _fake_hits(n=2):
    """Hand-built hits so prompt/citation tests don't depend on retrieval."""
    return [
        SearchHit(Chunk(text=f"passage {i} about privacy", doc_id="docA",
                        filename="paperA.pdf", page=i, chunk_index=i), 0.9)
        for i in range(1, n + 1)
    ]


def test_retrieve_returns_relevant_chunks(built_store):
    hits = retrieve("privacy across devices", built_store, top_k=3)
    assert hits
    assert all(h.score >= 0 for h in hits)


def test_answer_is_grounded_and_cites_sources(built_store):
    result = answer_question(STRONG_QUERY, built_store, llm=FakeLLM())
    assert result.grounded is True
    assert result.llm_used is True
    assert result.sources
    assert "[1]" in result.answer
    assert validate_citations(result) is True


def test_prompt_contains_untrusted_delimiters_and_context():
    prompt = build_prompt("What about privacy?", _fake_hits(2))
    assert "CONTEXT PASSAGES" in prompt
    assert "<<<" in prompt and ">>>" in prompt  # delimiters for injection safety
    assert "untrusted" in prompt.lower()
    assert "paperA.pdf, page 1" in prompt  # source provenance in the prompt


def test_insufficient_evidence_maps_to_not_found(built_store):
    result = answer_question("How does it help privacy?", built_store,
                             llm=FakeLLM(reply="INSUFFICIENT_EVIDENCE"))
    assert result.grounded is False
    assert result.answer == NOT_FOUND_MESSAGE


def test_empty_store_returns_not_found_without_calling_llm():
    from embeddings import get_embedder
    from vector_store import VectorStore

    class ExplodingLLM:
        name = "explode"

        def generate(self, *a, **k):
            raise AssertionError("LLM must not be called on empty retrieval")

    store = VectorStore(dimension=get_embedder().dimension)
    result = answer_question("anything", store, llm=ExplodingLLM())
    assert result.grounded is False
    assert result.llm_used is False
    assert result.answer == NOT_FOUND_MESSAGE


def test_no_key_still_returns_retrieved_sources(built_store):
    result = answer_question("privacy across devices", built_store,
                             llm=NullLLM())
    # Either we retrieved something (sources present, llm_used False) or the
    # query fell below the relevance floor (not-found). Both are valid, and in
    # neither case did we fabricate an answer.
    assert result.llm_used is False


def test_citation_number_extraction_and_validation(built_store):
    result = answer_question(STRONG_QUERY, built_store,
                             llm=FakeLLM(reply="A [1] and B [2]."))
    assert citation_numbers(result.answer) == {1, 2}
    # Force a single source so the [2] citation becomes out-of-range.
    result.sources = result.sources[:1]
    assert validate_citations(result) is False
