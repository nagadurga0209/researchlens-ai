# ResearchLens AI — Architecture

## High-level flow

```
            ┌──────────────────────────────────────────────────────────┐
            │                     Streamlit UI (app.py)                  │
            │  Upload · Chat · Summary · Compare · Gaps · History tabs   │
            └───────────────┬───────────────────────────┬──────────────┘
                            │                           │
                   upload PDF                    ask question
                            │                           │
                            ▼                           ▼
        ┌───────────────────────────┐      ┌───────────────────────────┐
        │   Ingestion (ingestion.py)│      │   RAG pipeline             │
        │                           │      │   (rag_pipeline.py)        │
        │  pdf_processor → chunking │      │                            │
        │  → embeddings → vector    │      │  embed query → retrieve    │
        │    store + storage (DB)   │      │  → build grounded prompt   │
        └───────┬─────────┬─────────┘      │  → LLM → answer + sources  │
                │         │                └───────┬──────────┬─────────┘
                ▼         ▼                        ▼          │
      ┌──────────────┐ ┌──────────────┐   ┌──────────────┐  │
      │ VectorStore  │ │  Storage     │   │ llm_providers│  │
      │ (FAISS /     │ │ (SQLite:     │   │ Gemini/OpenAI│  │
      │  NumPy)      │ │  docs+history│   │  /NullLLM    │  │
      │  + metadata  │ └──────────────┘   └──────────────┘  │
      │  persisted   │◄──────── retrieve chunks ────────────┘
      └──────────────┘
                ▲
                │  same embedder for docs AND queries
                │
        ┌──────────────┐
        │ embeddings.py│  SentenceTransformers (MiniLM, 384-d)
        │              │  → auto NumPy "hashing" fallback when offline
        └──────────────┘

   Analysis (paper_analysis.py): summaries / comparison / gap analysis
   read chunks from VectorStore and call llm_providers.

   Evaluation (evaluation.py + eval/): labelled questions → retrieve →
   Precision@K, Recall@K, MRR, latency, citation coverage.
```

## The RAG pipeline, step by step

1. **Ingest** — `pdf_processor` extracts text per page (preserving page
   numbers); `chunking` splits each page into overlapping, word-aware chunks;
   `embeddings` turns each chunk into a 384-dim vector; `vector_store` adds the
   vectors to a FAISS `IndexFlatIP`; `storage` records the document.
2. **Persist** — the FAISS index and a JSON metadata sidecar are written to
   `data/index/`, and the document row to SQLite, so indexing survives restart.
3. **Retrieve** — a question is embedded with the **same** model, FAISS returns
   the top-K most similar chunks (cosine similarity, because vectors are
   L2-normalized), optionally filtered to the papers the user selected.
4. **Generate** — the retrieved passages are wrapped in a grounded prompt and
   sent to the LLM, which must answer only from them, cite `[n]` passages, and
   emit `INSUFFICIENT_EVIDENCE` when the context can't support an answer.
5. **Answer with sources** — the UI shows the answer plus expandable source
   panels (filename, page, similarity score, passage text), and validates that
   every `[n]` citation refers to a real retrieved passage.

## Why these choices

| Component | Choice | Reason |
|-----------|--------|--------|
| Embeddings | `all-MiniLM-L6-v2` (384-d) | Small, fast, strong general-purpose semantic model; same model for query + docs keeps them in one space. |
| Index | FAISS `IndexFlatIP` | Exact search; with normalized vectors, inner product = cosine similarity. Right for a modest paper library. |
| Relevance score | cosine in [-1, 1] | Interpretable; a floor (`MIN_RELEVANCE`) filters weak matches so we can say "not found". |
| Metadata | JSON sidecar + SQLite | Vectors and bookkeeping kept separate and simple; content-hash `doc_id` enables dedup and clean deletes. |
| LLM | Pluggable (Gemini default) | `llm_providers.get_llm()` hides the vendor; swap by adding one class. |
| Fallback embedder/index | NumPy | Keeps the whole pipeline runnable and testable offline; real libs used automatically when installed. |

## Security / safety notes

- Retrieved paper text is treated as **untrusted data**. The prompt instructs
  the model to ignore any instructions embedded in passages, and each passage
  is fenced with `<<< >>>` delimiters.
- API keys live only in environment variables / `.env` (git-ignored). The app
  never fabricates an answer without a configured key — it shows the retrieved
  passages and a clear configuration message instead.
