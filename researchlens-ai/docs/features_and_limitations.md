# Completed features & known limitations

## Completed features

- [x] **Multi-PDF upload & indexing** — page-wise PyMuPDF extraction, filename
      and page preserved, overlapping chunks, embeddings, FAISS storage, live
      status, dedup by content hash, robust handling of empty/corrupt/non-PDF
      files.
- [x] **RAG question answering** — same-model query embedding, top-K retrieval,
      grounded prompt, answer with source filenames + page numbers + passages,
      honest "insufficient evidence" when unsupported.
- [x] **Chat interface** — Streamlit UI with sidebar upload/manage, document
      selection, suggested questions, chat history, loading indicators, error
      messages, expandable source panels.
- [x] **Summarization** — structured (problem/objectives/method/data/findings/
      limitations/future work), hierarchical for long papers, STATED vs
      INTERPRETATION labels.
- [x] **Multi-paper comparison** — structured markdown table across standard
      dimensions, "Not found in the provided document" for gaps.
- [x] **Research gap analysis** — common challenges, repeated limitations,
      conflicting findings, underexplored topics, potential directions, with
      source hints and a clear "potential, not verified" disclaimer.
- [x] **Evaluation** — Precision@K, Recall@K, MRR, latency, citation coverage;
      labelled dataset; experiments across chunk size / overlap / top-K; results
      saved to CSV.
- [x] **Persistence** — FAISS index + metadata + SQLite survive restarts;
      deletion removes a document from future retrieval.
- [x] **Pluggable LLM** — Gemini default, OpenAI-ready, NullLLM fallback.
- [x] **Tests** — 36-test pytest suite with mocked LLM calls.
- [x] **Security** — untrusted-content handling, env-var secrets.

## Known limitations

- **Scanned PDFs** without a text layer aren't supported (no OCR). The app
  detects this and reports it rather than indexing empty text.
- **Flat FAISS index** is exact but O(N) per query; fine for a personal paper
  library, not for millions of chunks. Swap to IVF/HNSW to scale.
- **Offline fallback embedder** (used only when sentence-transformers isn't
  installed) does lexical, not semantic, matching — install the requirements
  for real semantic search.
- **Chunking is fixed-size**, not section-aware; a heading-aware splitter would
  improve summaries and comparisons.
- **No re-ranker** after retrieval; adding a cross-encoder re-ranker would
  improve precision.
- **LLM output quality** depends on the chosen provider/model and your API
  quota; the pipeline grounds and cites but cannot guarantee a model never
  errs — hence the visible source passages.
- **Single-user, local** app; no authentication or multi-user isolation.
