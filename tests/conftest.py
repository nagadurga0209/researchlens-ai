"""Shared pytest fixtures for the ResearchLens AI test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project modules importable from within tests/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chunking import chunk_pages          # noqa: E402
from embeddings import get_embedder, reset_embedder  # noqa: E402
from vector_store import VectorStore      # noqa: E402


SAMPLE_PAGES = [
    (1, "Federated learning trains a shared model across many devices without "
        "moving raw data off the device, which improves user privacy."),
    (2, "Secret image sharing splits a secret image into shadow images using "
        "digit-wise decomposition to reduce the shadow size."),
    (3, "The evaluation uses MNIST and CIFAR-10 datasets and reports top-1 "
        "accuracy and the number of communication rounds."),
]


@pytest.fixture(autouse=True)
def _fresh_embedder():
    """Reset the cached embedder around each test for isolation."""
    reset_embedder()
    yield
    reset_embedder()


@pytest.fixture
def sample_pages():
    return list(SAMPLE_PAGES)


@pytest.fixture
def built_store(sample_pages):
    """A VectorStore indexed with the sample pages (one document)."""
    emb = get_embedder()
    store = VectorStore(dimension=emb.dimension)
    chunks = chunk_pages(sample_pages, doc_id="docA", filename="paperA.pdf",
                         chunk_size=400, overlap=80)
    store.add(emb.encode([c.text for c in chunks]), chunks)
    return store


class FakeLLM:
    """A deterministic fake LLM for testing pipeline logic without a network.

    Returns a grounded-looking answer that cites passage [1], unless the prompt
    asks it to refuse.
    """

    name = "fake-llm"

    def __init__(self, reply: str | None = None):
        self._reply = reply

    def generate(self, prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 1024) -> str:
        if self._reply is not None:
            return self._reply
        return "Based on the evidence, the method improves privacy [1]."


@pytest.fixture
def fake_llm():
    return FakeLLM()
