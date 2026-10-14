"""Central configuration for ResearchLens AI.

All tunable settings live here and are read from environment variables
(loaded from a local .env file via python-dotenv) with sensible defaults,
so the app runs out of the box and secrets never get hardcoded.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if present.
# Values already set in the real environment take precedence.
load_dotenv()


# --- Paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

# Persisted FAISS index + sidecar metadata.
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

# SQLite database for document records and chat history.
DB_PATH = BASE_DIR / "researchlens.db"

# Make sure the directories exist on import.
for _d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- LLM provider ----------------------------------------------------------
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


# --- Embeddings ------------------------------------------------------------
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
).strip()


# --- Chunking / retrieval --------------------------------------------------
def _int_env(name: str, default: int) -> int:
    """Read an int from the environment, falling back on bad/empty values."""
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


CHUNK_SIZE = _int_env("CHUNK_SIZE", 800)          # characters per chunk
CHUNK_OVERLAP = _int_env("CHUNK_OVERLAP", 150)    # overlap between chunks
TOP_K = _int_env("TOP_K", 5)                      # default retrieved chunks


def llm_is_configured() -> bool:
    """True when the selected provider has a usable API key.

    The app stays fully functional for ingestion/retrieval without this;
    only answer *generation* needs a real key.
    """
    if LLM_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    if LLM_PROVIDER == "openai":
        return bool(OPENAI_API_KEY)
    return False


def missing_key_message() -> str:
    """A clear, actionable message when no LLM key is configured."""
    if LLM_PROVIDER == "gemini":
        return (
            "No Gemini API key found. Set GEMINI_API_KEY in your .env file. "
            "Get a free key at https://aistudio.google.com/app/apikey. "
            "Retrieval and source passages still work without it — only "
            "AI-written answers are disabled."
        )
    if LLM_PROVIDER == "openai":
        return (
            "No OpenAI API key found. Set OPENAI_API_KEY in your .env file. "
            "Retrieval and source passages still work without it — only "
            "AI-written answers are disabled."
        )
    return (
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Set it to 'gemini' or "
        f"'openai' in your .env file."
    )
