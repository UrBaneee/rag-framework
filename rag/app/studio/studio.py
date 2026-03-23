"""RAG Studio — Streamlit multi-page application entry point.

Run with:
    streamlit run rag/app/studio/studio.py

Pages are auto-discovered from the ``pages/`` directory by Streamlit's
multi-page app convention (files prefixed with a number determine order).
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Studio",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Home page content
# ---------------------------------------------------------------------------

st.title("🔍 RAG Studio")
st.markdown(
    """
    Welcome to **RAG Studio** — a developer workbench for the RAG framework.

    Use the sidebar to navigate between pages:

    | Page | Description |
    |------|-------------|
    | **1 · Ingestion Manager** | Upload documents and run the ingest pipeline |
    | **2 · Ingestion Traces** | Inspect per-run parse, route, and chunk events |
    | **3 · Query Traces** | Run queries and inspect retrieval, reranking, and generation |
    | **4 · Evaluation Panel** | Run and review RAG evaluation metrics |
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.info(
        "**Getting started**\n\n"
        "1. Go to **Ingestion Manager** and upload a document.\n"
        "2. Open **Query Traces** to ask questions against your corpus.\n"
        "3. Check **Ingestion Traces** for pipeline diagnostics.",
        icon="💡",
    )
with col2:
    st.info(
        "**About**\n\n"
        "RAG Studio is built on a modular RAG framework with pluggable "
        "parsers, cleaners, chunkers, embedders, and LLM clients.\n\n"
        "All pipeline runs are recorded in the TraceStore for full observability.",
        icon="ℹ️",
    )
