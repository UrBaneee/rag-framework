"""Ingestion Traces page — inspect parse, route, and chunk events by run_id."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Ingestion Traces · RAG Studio", layout="wide")

st.title("🔬 Ingestion Traces")
st.caption("Inspect per-run parse, route, clean, and chunk pipeline events.")

# ---------------------------------------------------------------------------
# Sidebar — DB path
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    db_path = st.text_input("SQLite DB path", value="data/rag.db")
    max_runs = st.number_input("Recent runs to show", min_value=5, max_value=200, value=50)

# ---------------------------------------------------------------------------
# Load runs from TraceStore
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def _load_runs(db: str, limit: int) -> list[dict]:
    """Load recent ingestion runs from the TraceStore (cached 10 s)."""
    try:
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        store = SQLiteTraceStore(db)
        # Ingest runs are recorded as "ingest" run_type entries
        all_runs = store.list_runs(limit=int(limit))
        return [r for r in all_runs if r["run_type"] in (
            "ingest", "ingest_start", "ingest_complete", "ingest_error",
        )]
    except Exception as exc:
        return []


@st.cache_data(ttl=10)
def _load_events_for_run(db: str, run_id: str) -> list[dict]:
    """Load all events that share the given run_id (or were created by that run)."""
    try:
        from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
        store = SQLiteTraceStore(db)
        all_runs = store.list_runs(limit=500)
        return [
            r for r in all_runs
            if r["run_id"] == run_id
            or r.get("metadata", {}).get("run_id") == run_id
        ]
    except Exception as exc:
        return []


# ---------------------------------------------------------------------------
# Main layout: two tabs
# ---------------------------------------------------------------------------

tab_browse, tab_detail = st.tabs(["📋 Browse Runs", "🔍 Run Detail"])

with tab_browse:
    if not Path(db_path).exists():
        st.warning(f"Database not found at `{db_path}`. Run an ingestion first.")
    else:
        runs = _load_runs(db_path, int(max_runs))
        if not runs:
            st.info("No ingestion runs found. Upload a document on the Ingestion Manager page first.")
        else:
            st.markdown(f"**{len(runs)}** recent ingestion run(s) found.")
            from rag.app.studio.components.trace_viewer import render_run_summary_table
            render_run_summary_table(runs)

with tab_detail:
    if not Path(db_path).exists():
        st.warning(f"Database not found at `{db_path}`. Run an ingestion first.")
    else:
        all_runs = _load_runs(db_path, int(max_runs))

        from rag.app.studio.components.trace_viewer import render_run_selector, render_run_events

        selected_run_id = render_run_selector(all_runs, label="Select ingestion run")

        if selected_run_id:
            st.markdown(f"**Run ID:** `{selected_run_id}`")
            st.divider()
            events = _load_events_for_run(db_path, selected_run_id)
            render_run_events(events)
