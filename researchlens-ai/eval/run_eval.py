"""Run the retrieval evaluation and print real metrics.

Usage (from the project root, with the venv active):
    python eval/run_eval.py

It builds an in-memory index from the small labelled corpus in eval/corpus/,
then evaluates retrieval across several (chunk_size, chunk_overlap, top_k)
settings and prints the per-query and summary tables. No LLM key required —
this measures *retrieval*, which is the part that determines answer quality.

The numbers you see are computed from actual retrieval output on your machine.
With the real sentence-transformers model installed they reflect true semantic
search; without it, the deterministic fallback embedder is used (lexical
overlap), so treat those numbers as a lower bound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# Make the project root importable when run as `python eval/run_eval.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chunking import chunk_pages              # noqa: E402
from embeddings import get_embedder            # noqa: E402
from evaluation import load_eval_items, run_evaluation  # noqa: E402
from vector_store import VectorStore           # noqa: E402

CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
QUESTIONS = Path(__file__).resolve().parent / "eval_questions.json"


def build_store(chunk_size: int, overlap: int) -> VectorStore:
    """Index the labelled corpus into a fresh in-memory store."""
    emb = get_embedder()
    store = VectorStore(dimension=emb.dimension)
    for path in sorted(CORPUS_DIR.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        pages = [(int(p), str(t)) for p, t in doc["pages"]]
        chunks = chunk_pages(pages, doc_id=doc["filename"],
                             filename=doc["filename"],
                             chunk_size=chunk_size, overlap=overlap)
        store.add(emb.encode([c.text for c in chunks]), chunks)
    return store


def main() -> None:
    emb = get_embedder()
    print(f"Embedder: {emb.name}  (fallback={emb.is_fallback})\n")
    items = load_eval_items(QUESTIONS)
    print(f"Loaded {len(items)} labelled questions.\n")

    experiments = [
        {"chunk_size": 400, "overlap": 80, "top_k": 3},
        {"chunk_size": 800, "overlap": 150, "top_k": 3},
        {"chunk_size": 800, "overlap": 150, "top_k": 5},
        {"chunk_size": 1200, "overlap": 200, "top_k": 5},
    ]

    summaries = []
    for exp in experiments:
        store = build_store(exp["chunk_size"], exp["overlap"])
        report = run_evaluation(items, store, top_k=exp["top_k"],
                                match_key="filename",
                                chunk_size=exp["chunk_size"],
                                chunk_overlap=exp["overlap"])
        s = report.summary()
        # record the experiment knobs with the result
        s_row = {**exp, **{k: s[k] for k in
                           ["mean_precision@k", "mean_recall@k", "MRR",
                            "mean_latency_ms", "citation_coverage"]}}
        summaries.append(s_row)

        print("=" * 72)
        print(f"Experiment: chunk_size={exp['chunk_size']} "
              f"overlap={exp['overlap']} top_k={exp['top_k']}")
        print("-" * 72)
        print(report.to_dataframe().to_string(index=False))
        print("\nSummary:", json.dumps(s, indent=2))
        print()

    print("=" * 72)
    print("ALL EXPERIMENTS — summary comparison")
    print("=" * 72)
    df = pd.DataFrame(summaries)
    print(df.to_string(index=False))

    out = Path(__file__).resolve().parent / "eval_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved comparison table to {out}")


if __name__ == "__main__":
    main()
