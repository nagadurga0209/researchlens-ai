"""ResearchLens AI — Streamlit web application.

Run with:  streamlit run app.py

The entire UI is built from Streamlit's built-in components. Heavy objects
(the vector store, the embedder, the database) are cached in st.session_state
so they survive reruns without being rebuilt on every interaction.

This file deliberately keeps *logic* in the library modules and *presentation*
here, so the app stays readable.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

import config
from embeddings import get_embedder
from ingestion import delete_document, ingest_pdf, load_or_create_store
from paper_analysis import compare_papers, research_gap_analysis, summarize_paper
from rag_pipeline import answer_question, validate_citations
from storage import Storage

logging.basicConfig(level=logging.INFO)

SUGGESTED_QUESTIONS = [
    "What problem does this paper try to solve?",
    "What methodology or algorithm is proposed?",
    "What datasets and evaluation metrics are used?",
    "What are the main results and limitations?",
]


# --- session bootstrap -----------------------------------------------------
def _init_state() -> None:
    if "storage" not in st.session_state:
        st.session_state.storage = Storage()
    if "store" not in st.session_state:
        st.session_state.store = load_or_create_store(st.session_state.storage)
    if "messages" not in st.session_state:
        # list of dicts: {role, content, sources?}
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def _docs_table() -> pd.DataFrame:
    docs = st.session_state.storage.list_documents()
    if not docs:
        return pd.DataFrame(columns=["filename", "num_pages", "num_chunks"])
    df = pd.DataFrame(docs)
    return df[["filename", "num_pages", "num_chunks", "added_at"]]


# --- sidebar: upload & manage ----------------------------------------------
def render_sidebar() -> list[str]:
    """Returns the list of selected doc_ids to scope queries to."""
    st.sidebar.title("📄 Documents")

    # Embedder status (warn if the fallback is active).
    emb = get_embedder()
    if emb.is_fallback:
        st.sidebar.warning(
            "Using the offline **fallback embedder** (no semantic model "
            "found). Install requirements for real semantic search.",
            icon="⚠️",
        )

    uploads = st.sidebar.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs. Identical files are not re-indexed.",
    )
    if uploads:
        for uploaded in uploads:
            data = uploaded.getvalue()
            with st.sidebar.status(f"Indexing {uploaded.name}…",
                                   expanded=False) as status:
                result = ingest_pdf(
                    data, uploaded.name,
                    st.session_state.store, st.session_state.storage,
                )
                if not result.ok:
                    status.update(label=f"❌ {uploaded.name}: {result.error}",
                                  state="error")
                elif result.skipped:
                    status.update(label=f"↷ {uploaded.name} already indexed",
                                  state="complete")
                else:
                    status.update(
                        label=(f"✅ {uploaded.name}: {result.num_pages} pages, "
                               f"{result.num_chunks} chunks"),
                        state="complete",
                    )

    st.sidebar.divider()

    docs = st.session_state.storage.list_documents()
    if not docs:
        st.sidebar.info("No documents indexed yet. Upload a PDF to begin.")
        return []

    st.sidebar.subheader("Select papers to query")
    id_by_label = {f"{d['filename']}": d["doc_id"] for d in docs}
    labels = list(id_by_label.keys())
    selected_labels = st.sidebar.multiselect(
        "Scope questions to these papers (none = all):",
        options=labels,
        default=labels,
    )
    selected_ids = [id_by_label[l] for l in selected_labels]

    # Delete controls.
    with st.sidebar.expander("Manage / delete"):
        st.dataframe(_docs_table(), hide_index=True, use_container_width=True)
        to_delete = st.selectbox("Delete a document",
                                 ["—"] + labels, index=0)
        if st.button("Delete", type="secondary") and to_delete != "—":
            doc_id = id_by_label[to_delete]
            removed = delete_document(doc_id, st.session_state.store,
                                      st.session_state.storage)
            st.success(f"Deleted '{to_delete}' ({removed} chunks removed).")
            st.rerun()

    return selected_ids


# --- source rendering ------------------------------------------------------
def render_sources(sources) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources ({len(sources)} passages)"):
        for i, hit in enumerate(sources, start=1):
            st.markdown(
                f"**[{i}] {hit.chunk.filename} — page {hit.chunk.page}** "
                f"· similarity {hit.score:.3f}"
            )
            st.caption(hit.chunk.text[:600]
                       + ("…" if len(hit.chunk.text) > 600 else ""))
            if i < len(sources):
                st.divider()


# --- chat tab --------------------------------------------------------------
def render_chat(selected_ids: list[str]) -> None:
    st.subheader("Ask about your papers")

    # Suggested questions as quick buttons.
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(cols, SUGGESTED_QUESTIONS):
        if col.button(q, use_container_width=True):
            st.session_state.pending_question = q

    # Replay the conversation.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_sources(msg["sources"])

    typed = st.chat_input("Ask a question about the selected paper(s)…")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None

    if not question:
        return

    if st.session_state.store.size == 0:
        st.warning("Upload and index at least one PDF first.")
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and composing an answer…"):
            scope = set(selected_ids) if selected_ids else None
            result = answer_question(
                question, st.session_state.store, allowed_doc_ids=scope,
            )
        st.markdown(result.answer)
        if result.note and not result.llm_used:
            st.info(result.note)
        if result.sources and not validate_citations(result):
            st.warning("⚠️ The answer cited a passage number that wasn't "
                       "provided; treat citations with caution.")
        render_sources(result.sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "sources": result.sources,
    })
    st.session_state.storage.add_history(
        question, result.answer, result.sources_as_text())


# --- analysis tabs ---------------------------------------------------------
def _doc_picker(label: str, multi: bool):
    docs = st.session_state.storage.list_documents()
    id_by_label = {d["filename"]: d["doc_id"] for d in docs}
    labels = list(id_by_label.keys())
    if not labels:
        st.info("Upload a paper first.")
        return [], id_by_label
    if multi:
        chosen = st.multiselect(label, labels, default=labels[:2])
    else:
        chosen = [st.selectbox(label, labels)]
    return [id_by_label[c] for c in chosen if c], id_by_label


def render_summary_tab() -> None:
    st.subheader("Paper summary")
    ids, _ = _doc_picker("Choose a paper to summarize", multi=False)
    if ids and st.button("Generate summary", key="sum_btn"):
        with st.spinner("Summarizing (hierarchical for long papers)…"):
            res = summarize_paper(ids[0], st.session_state.store)
        st.markdown(f"### {res.title}")
        if not res.llm_used:
            st.info(res.note or "Summarization requires a configured LLM.")
        st.markdown(res.body)


def render_compare_tab() -> None:
    st.subheader("Compare papers")
    ids, _ = _doc_picker("Choose two or more papers", multi=True)
    if len(ids) >= 2 and st.button("Compare", key="cmp_btn"):
        with st.spinner("Comparing papers…"):
            res = compare_papers(ids, st.session_state.store)
        st.markdown(f"### {res.title}")
        if not res.llm_used:
            st.info(res.note or "Comparison requires a configured LLM.")
        st.markdown(res.body)
    elif ids and len(ids) < 2:
        st.caption("Select at least two papers.")


def render_gap_tab() -> None:
    st.subheader("Research gap analysis")
    ids, _ = _doc_picker("Choose one or more papers", multi=True)
    if ids and st.button("Analyze gaps", key="gap_btn"):
        with st.spinner("Analyzing research gaps…"):
            res = research_gap_analysis(ids, st.session_state.store)
        st.markdown(f"### {res.title}")
        if not res.llm_used:
            st.info(res.note or "Gap analysis requires a configured LLM.")
        st.markdown(res.body)


def render_history_tab() -> None:
    st.subheader("Question history")
    history = st.session_state.storage.list_history()
    if not history:
        st.info("No questions asked yet.")
        return
    for h in history:
        with st.expander(f"Q: {h['question']}  ·  {h['created_at']}"):
            st.markdown(h["answer"])
            if h["sources"]:
                st.caption(f"Sources: {h['sources']}")
    if st.button("Clear history"):
        st.session_state.storage.clear_history()
        st.rerun()


# --- main ------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="ResearchLens AI", page_icon="🔬",
                       layout="wide")
    _init_state()

    st.title("🔬 ResearchLens AI")
    st.caption("Upload research papers and ask grounded, cited questions — a "
               "Retrieval-Augmented Generation assistant.")

    # Global config banner if the LLM key is missing.
    if not config.llm_is_configured():
        st.warning(config.missing_key_message(), icon="🔑")

    selected_ids = render_sidebar()

    tab_chat, tab_sum, tab_cmp, tab_gap, tab_hist = st.tabs(
        ["💬 Chat", "📝 Summary", "⚖️ Compare", "🧭 Gaps", "🕘 History"]
    )
    with tab_chat:
        render_chat(selected_ids)
    with tab_sum:
        render_summary_tab()
    with tab_cmp:
        render_compare_tab()
    with tab_gap:
        render_gap_tab()
    with tab_hist:
        render_history_tab()


if __name__ == "__main__":
    main()
