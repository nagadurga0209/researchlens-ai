"""Retrieval evaluation for ResearchLens AI.

Given a small set of questions, each labelled with the document+page(s) that
genuinely answer it ("relevant" items), we measure how well the retriever
surfaces those items. This is honest, reproducible evaluation — the metrics
are computed from real retrieval output, never hand-written.

Metrics implemented:
  * Precision@K  — of the K retrieved, what fraction are relevant.
  * Recall@K     — of all relevant items, what fraction appear in the top K.
  * MRR          — mean reciprocal rank of the first relevant hit.
  * Latency      — wall-clock seconds per query.
  * Citation coverage — fraction of queries where at least one retrieved chunk
                        is relevant (i.e. a citable source exists for the
                        answer). Defined explicitly so it isn't ambiguous.

A "relevant" retrieved chunk is one whose (doc_id, page) matches one of the
labelled relevant (doc_id, page) pairs for that question.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config
from rag_pipeline import retrieve
from vector_store import VectorStore


@dataclass
class EvalItem:
    """One labelled evaluation question."""

    question: str
    # Relevant locations as (doc_id_or_filename, page) pairs.
    relevant: list[tuple[str, int]]
    note: str = ""


@dataclass
class QueryMetrics:
    question: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    latency_s: float
    has_citation: bool
    retrieved: list[tuple[str, int]] = field(default_factory=list)


def _hit_key(hit, key: str) -> tuple[str, int]:
    """Make a (identifier, page) key from a SearchHit for matching.

    Labels may use either the doc_id or the filename, so we support both.
    """
    ident = hit.chunk.doc_id if key == "doc_id" else hit.chunk.filename
    return (ident, hit.chunk.page)


def evaluate_item(
    item: EvalItem,
    store: VectorStore,
    top_k: int = config.TOP_K,
    match_key: str = "filename",
) -> QueryMetrics:
    """Run one query and compute its metrics."""
    relevant = set(item.relevant)

    t0 = time.perf_counter()
    hits = retrieve(item.question, store, top_k=top_k)
    latency = time.perf_counter() - t0

    retrieved_keys = [_hit_key(h, match_key) for h in hits]
    retrieved_set = set(retrieved_keys)

    num_relevant_retrieved = len(retrieved_set & relevant)
    precision = num_relevant_retrieved / len(hits) if hits else 0.0
    recall = num_relevant_retrieved / len(relevant) if relevant else 0.0

    rr = 0.0
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in relevant:
            rr = 1.0 / rank
            break

    return QueryMetrics(
        question=item.question,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=rr,
        latency_s=latency,
        has_citation=num_relevant_retrieved > 0,
        retrieved=retrieved_keys,
    )


@dataclass
class EvalReport:
    per_query: list[QueryMetrics]
    top_k: int
    chunk_size: int
    chunk_overlap: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "question": m.question[:60],
                "precision@k": round(m.precision_at_k, 3),
                "recall@k": round(m.recall_at_k, 3),
                "RR": round(m.reciprocal_rank, 3),
                "latency_ms": round(m.latency_s * 1000, 1),
                "cited": m.has_citation,
            }
            for m in self.per_query
        ])

    def summary(self) -> dict:
        n = len(self.per_query) or 1
        return {
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "num_queries": len(self.per_query),
            "mean_precision@k": round(
                sum(m.precision_at_k for m in self.per_query) / n, 3),
            "mean_recall@k": round(
                sum(m.recall_at_k for m in self.per_query) / n, 3),
            "MRR": round(
                sum(m.reciprocal_rank for m in self.per_query) / n, 3),
            "mean_latency_ms": round(
                sum(m.latency_s for m in self.per_query) / n * 1000, 1),
            "citation_coverage": round(
                sum(1 for m in self.per_query if m.has_citation) / n, 3),
        }


def run_evaluation(
    items: list[EvalItem],
    store: VectorStore,
    top_k: int = config.TOP_K,
    match_key: str = "filename",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> EvalReport:
    """Evaluate all items. ``chunk_size``/``chunk_overlap`` are recorded in the
    report for reproducibility; pass the values actually used to build the
    store (defaults fall back to the global config)."""
    per_query = [evaluate_item(it, store, top_k=top_k, match_key=match_key)
                 for it in items]
    return EvalReport(
        per_query,
        top_k,
        chunk_size if chunk_size is not None else config.CHUNK_SIZE,
        chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
    )


def load_eval_items(path: Path) -> list[EvalItem]:
    """Load an evaluation set from JSON.

    Expected shape:
        [{"question": "...", "relevant": [["file.pdf", 2], ...], "note": ""}]
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = []
    for r in raw:
        rel = [(str(a), int(b)) for a, b in r["relevant"]]
        items.append(EvalItem(r["question"], rel, r.get("note", "")))
    return items
