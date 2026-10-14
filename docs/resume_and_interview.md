# Resume bullets & interview preparation

## Resume-ready bullets (based only on implemented features)

Pick 2–4 that fit your resume's length. Each maps to code that actually exists
in this project.

- Built **ResearchLens AI**, a Retrieval-Augmented Generation (RAG) assistant
  in Python/Streamlit that lets users query, summarize, and compare multiple
  research papers with answers grounded in the source text and cited to
  filename + page.
- Implemented the full RAG pipeline end to end: PyMuPDF PDF extraction →
  overlapping word-aware chunking → Sentence-Transformers (`all-MiniLM-L6-v2`,
  384-d) embeddings → **FAISS** `IndexFlatIP` cosine retrieval → LLM answer
  generation with inline `[n]` citations.
- Engineered **anti-hallucination** safeguards: a cosine-similarity relevance
  floor, an explicit "insufficient evidence" path, and automatic validation
  that every citation maps to a real retrieved passage.
- Persisted the FAISS index + JSON metadata and a SQLite store of documents
  and chat history so indexed papers and history survive restarts; used
  content-hash document IDs to deduplicate uploads and cleanly delete papers
  from future retrieval.
- Designed a **pluggable LLM layer** (Gemini default, OpenAI-ready) so the
  provider can be swapped without touching the pipeline, with graceful
  degradation when no API key is configured.
- Wrote a **retrieval evaluation harness** (Precision@K, Recall@K, MRR,
  latency, citation coverage) with a labelled question set, and ran experiments
  across chunk sizes and top-K settings; achieved recall@5 = 0.9 on the
  evaluation set.
- Covered the system with a **36-test pytest suite** (chunking, extraction,
  embeddings, FAISS retrieval, deletion, citation validation, metrics) using
  mocked LLM calls.

> Tip: quote metrics you actually reproduce on your own machine with the real
> model — rerun `python eval/run_eval.py` and use those numbers.

---

## Interview Q&A

### RAG (Retrieval-Augmented Generation)

**Q: What is RAG and why use it here?**
RAG couples a retriever with a generator: instead of asking an LLM to answer
from its parameters alone, you first fetch relevant passages from your own
corpus and put them in the prompt. For research papers this means answers are
grounded in the actual documents, you get citations, you can use papers the
model never saw in training, and you avoid retraining when documents change.

**Q: Walk me through your pipeline.**
PDF → page-wise text extraction (PyMuPDF) → overlapping chunks → embed each
chunk with MiniLM → store vectors in FAISS with page/filename metadata. At
query time: embed the question with the *same* model → FAISS returns the top-K
most similar chunks → wrap them in a grounded prompt → the LLM answers using
only those passages and cites them → the UI shows the answer plus the source
passages.

### Embeddings

**Q: What is an embedding?**
A dense vector that places text in a space where semantically similar text is
close together. We use `all-MiniLM-L6-v2`, which outputs 384-dimensional
vectors. Similar meaning → small angle between vectors → high cosine
similarity.

**Q: Why use the same model for documents and queries?**
Similarity is only meaningful if both live in the same vector space. Different
models produce incompatible spaces, so query–document distances would be
noise. We enforce one shared embedder in `embeddings.get_embedder()`.

### FAISS & retrieval

**Q: What is FAISS and which index did you choose?**
FAISS is a library for fast similarity search over dense vectors. I used
`IndexFlatIP` — a "flat" (exact, brute-force) index using inner product.
Because I L2-normalize every vector, the inner product equals cosine
similarity. Flat is exact and needs no tuning, which is ideal for a paper
library of this size; for millions of vectors you'd switch to an approximate
index like IVF or HNSW to trade a little recall for speed.

**Q: How is the relevance score computed?**
Cosine similarity in [-1, 1]; higher = more relevant. I apply a floor
(`MIN_RELEVANCE`) so weak matches are dropped and the system can honestly say
"not found" rather than retrieving irrelevant text.

**Q: How do you handle deletion in a flat index?**
A flat index has no stable per-vector delete, so I rebuild the index from the
surviving vectors and drop the deleted document's metadata. Correct and simple
at this scale.

### Chunking

**Q: Why chunk, and why overlap?**
Embedding a whole paper into one vector loses detail and blows past model input
limits; chunking gives fine-grained, retrievable units. Overlap (I use ~150
chars) means an idea split across a boundary still appears whole in one chunk,
so retrieval doesn't miss it. Chunking is word-aware so we never cut a word.

**Q: How did you pick chunk size / overlap / K?**
Empirically — the evaluation harness sweeps settings and reports Recall@K,
MRR, etc. Bigger chunks give more context but coarser matches; larger K raises
recall but can lower precision. I chose defaults (800/150, K=5) from those
results.

### Hallucination & citations

**Q: How do you reduce hallucination?**
Four layers: (1) the prompt restricts the model to the provided passages and
forbids outside knowledge; (2) a relevance floor + an explicit
`INSUFFICIENT_EVIDENCE` path means no evidence → no invented answer; (3)
citations are validated so a cited `[n]` must map to a real passage; (4) the
user always sees the source passages to verify the answer themselves.

**Q: How do citations work?**
Passages are numbered `[1..K]` in the prompt; the model cites those numbers;
`validate_citations()` checks every `[n]` in the answer is within range. The UI
links each number to its filename, page, and text.

**Q: What about prompt injection from a malicious PDF?**
Retrieved text is treated as untrusted data: it's fenced with delimiters and
the system prompt tells the model to ignore any instructions inside passages
and use them only as information to analyze.

### Evaluation

**Q: How do you know retrieval is any good?**
A labelled set of questions, each tagged with the document+page that truly
answers it. For each query I compute Precision@K (fraction of retrieved that
are relevant), Recall@K (fraction of relevant that were retrieved), MRR (how
high the first relevant hit ranks), latency, and citation coverage (fraction of
questions with at least one relevant retrieved passage). All numbers come from
real retrieval output — nothing is hand-written.

**Q: Precision vs recall here?**
Precision = of what I returned, how much was right; recall = of what was right,
how much I returned. With one relevant chunk per question, precision@K is
capped at 1/K, so I read it alongside recall and MRR rather than alone.

### System design

**Q: How does it persist state?**
FAISS index + JSON metadata on disk, plus SQLite for document records and
history. On startup the store is reloaded, so previously indexed papers are
immediately searchable.

**Q: How would you scale it?**
Swap `IndexFlatIP` for an approximate index (IVF/HNSW); batch and cache
embeddings; move SQLite to Postgres and the index to a managed vector DB; add a
re-ranker after retrieval; and chunk by section headings rather than fixed
size.
