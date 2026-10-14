# 🔬 ResearchLens AI — Research Paper RAG Assistant

Upload research papers (PDF), ask questions, and get answers **grounded in the
documents and cited to filename + page**. Summarize papers, compare several at
once, and surface potential research gaps — all in a single Python + Streamlit
app. No full-stack knowledge required.

Built to demonstrate **Generative AI, Retrieval-Augmented Generation (RAG),
embeddings, vector databases (FAISS), LLMs, and semantic search.**

---

## ✨ Features

| Feature | What it does |
|---|---|
| 📥 PDF upload & indexing | Extracts text page-by-page, chunks it, embeds it, stores it in FAISS. Handles bad files; skips duplicates. |
| 💬 RAG Q&A | Retrieves relevant passages and answers with inline `[n]` citations + source panels. Says "not found" instead of guessing. |
| 📝 Summaries | Structured, hierarchical summaries of long papers; separates stated facts from interpretation. |
| ⚖️ Comparison | Side-by-side table across problem, method, datasets, metrics, results, limitations, future work. |
| 🧭 Gap analysis | Common challenges, conflicts, underexplored topics, and *potential* directions (clearly labelled). |
| 📊 Evaluation | Precision@K, Recall@K, MRR, latency, citation coverage across chunk/top-K settings. |

See [`docs/features_and_limitations.md`](docs/features_and_limitations.md) for
the full list and known limitations, and
[`docs/architecture.md`](docs/architecture.md) for the design.

---

## 🧱 Tech stack

Python · Streamlit · PyMuPDF · Sentence-Transformers (`all-MiniLM-L6-v2`) ·
FAISS · Google Gemini (pluggable; OpenAI-ready) · SQLite · Pandas · Pytest.

---

## 🚀 Setup & run (Windows PowerShell)

> Prerequisite: **Python 3.10–3.12** installed and on your PATH
> (`python --version`).

```powershell
# 1. Go to the project folder
cd researchlens-ai

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
.\venv\Scripts\Activate.ps1
# If you get an execution-policy error, run this once in the same window:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# then re-run the Activate line.

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Configure your LLM API key
Copy-Item .env.example .env
#   Open .env in a text editor and paste your key into GEMINI_API_KEY=
#   Get a free Gemini key at https://aistudio.google.com/app/apikey

# 6. Run the app
streamlit run app.py
#   Your browser opens at http://localhost:8501

# 7. Run the tests (optional)
pip install pytest        # already in requirements.txt
pytest -q

# 8. Run the retrieval evaluation (optional, no API key needed)
python eval\run_eval.py
```

macOS / Linux is the same, but activate with `source venv/bin/activate`.

### First run note
On first launch the embedding model (~90 MB) downloads automatically and is
cached. This needs an internet connection once; afterwards it works offline.

---

## 🔑 Configuration

All settings live in `.env` (copied from `.env.example`). Key ones:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai`. |
| `GEMINI_API_KEY` | *(empty)* | Your Gemini key. Required for AI answers. |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model for generation. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model (docs **and** queries). |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Chunking (characters). |
| `TOP_K` | `5` | Chunks retrieved per question. |

**Without an API key** the app still runs: upload, indexing, retrieval, and
source passages all work — only AI-written answers are disabled, and the app
tells you so. It never fabricates an answer without credentials.

To use OpenAI instead: set `LLM_PROVIDER=openai`, `OPENAI_API_KEY=...`, and
`pip install openai`.

---

## 📁 Project structure

```
researchlens-ai/
├── app.py                # Streamlit UI (presentation only)
├── config.py             # Settings from .env
├── pdf_processor.py      # PyMuPDF extraction + validation + content hash
├── chunking.py           # Overlapping, word-aware chunking
├── embeddings.py         # Shared embedder (MiniLM + offline fallback)
├── vector_store.py       # FAISS index (+ NumPy fallback), metadata, persistence
├── storage.py            # SQLite: documents + chat history
├── ingestion.py          # Orchestrates PDF → indexed doc; dedup & delete
├── rag_pipeline.py       # Retrieve → grounded prompt → answer + citations
├── paper_analysis.py     # Summaries, comparison, gap analysis
├── llm_providers.py      # Pluggable LLM (Gemini / OpenAI / NullLLM)
├── evaluation.py         # Precision@K, Recall@K, MRR, latency, coverage
├── requirements.txt
├── .env.example          # Copy to .env (never commit .env)
├── .gitignore
├── data/
│   ├── uploads/          # saved PDFs (git-ignored)
│   └── index/            # FAISS index + metadata (git-ignored)
├── eval/
│   ├── corpus/           # small labelled mini-corpus (JSON)
│   ├── eval_questions.json
│   └── run_eval.py       # prints real retrieval metrics
├── tests/                # pytest suite (36 tests)
└── docs/
    ├── architecture.md
    ├── sample_questions.md
    ├── resume_and_interview.md
    └── features_and_limitations.md
```

---

## 🧪 How it works (short version)

PDFs → text extraction → overlapping chunks → embeddings → **FAISS** → retrieve
top-K → LLM → **answer with sources**. Vectors are L2-normalized, so FAISS inner
product = cosine similarity (relevance score in [-1, 1]). The question uses the
**same** embedding model as the documents. If retrieval finds nothing relevant,
the app says so rather than inventing an answer. Full details in
[`docs/architecture.md`](docs/architecture.md).

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Activate.ps1 cannot be loaded … execution policy` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window, then activate again. |
| `pip install faiss-cpu` fails | Ensure 64-bit Python 3.10–3.12 (FAISS has no 3.13 wheels yet on Windows). Upgrade pip: `python -m pip install --upgrade pip`. As a fallback the app still runs with a NumPy index. |
| Dependency/version conflicts | Use a **fresh** venv (delete `venv/` and recreate). Don't mix with a global install. |
| First run is slow / model download fails | The MiniLM model (~90 MB) downloads once; check your connection. Until it's available the app auto-uses the offline fallback embedder (lexical, lower quality) and shows a warning. |
| `No Gemini API key found` banner | Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. Get a free key at https://aistudio.google.com/app/apikey. |
| PDF indexes with 0 chunks / "no extractable text" | The PDF is likely scanned images; this app has no OCR. Use a text-based PDF. |
| Port 8501 in use | `streamlit run app.py --server.port 8502`. |
| Want to reset everything | Delete `data/index/*`, `data/uploads/*`, and `researchlens.db`. |

---

## 📄 License & attribution
Personal/portfolio project. The embedding model and LLM providers are subject
to their own licenses and terms.
