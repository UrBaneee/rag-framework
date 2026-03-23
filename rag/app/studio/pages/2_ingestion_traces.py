"""Ingestion Traces page — inspect parse, route, and chunk events by run_id."""

import streamlit as st

st.set_page_config(page_title="Ingestion Traces · RAG Studio", layout="wide")

st.title("🔬 Ingestion Traces")
st.caption("Inspect per-run parse, route, clean, and chunk pipeline events.")

st.info(
    "**Coming soon** — full trace viewer (Task 8.3).\n\n"
    "This page will let you select an ingestion run by ID, then inspect "
    "each pipeline stage: sniffer output, parser selection, quality gate "
    "result, cleaner steps, block count, chunk count, and embedding stats.",
    icon="🚧",
)

with st.expander("What is a trace?", expanded=False):
    st.markdown(
        """
        Every ingest pipeline run records a **trace** in the TraceStore (SQLite).
        Each trace captures:

        - `run_id` — unique identifier for the run
        - `source_path` — the file that was ingested
        - `run_type` — stage name (e.g. `ingest_start`, `parse_complete`, …)
        - `metadata` — stage-specific JSON payload (block count, token usage, …)
        - `created_at` — timestamp

        Use this page to drill into any run and understand exactly how each
        document was processed.
        """
    )
