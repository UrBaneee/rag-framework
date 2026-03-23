"""Evaluation Panel — run and review RAG evaluation suites.

Supports:
- example_queries suite (representative grounded questions)
- failure_cases suite (ambiguous / out-of-domain / noisy queries)

Metrics displayed: Recall@K, MRR, nDCG@K, source attribution,
efficiency (latency, token savings).  Tooltips come from
``metric_glossary.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Evaluation Panel · RAG Studio", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent.parent.parent / "tests" / "fixtures"

_SUITES = {
    "example_queries": _FIXTURES_DIR / "example_queries.json",
    "failure_cases": _FIXTURES_DIR / "failure_cases.json",
}

_DEFAULT_DB = "data/default.db"
_DEFAULT_INDEX = "data/default_indexes"


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "eval_report": None,
        "eval_suite": None,
        "eval_running": False,
        "eval_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_suite(path: Path) -> list[dict]:
    """Load a JSON or JSONL suite file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _run_evaluation(
    suite_name: str,
    db_path: str,
    index_dir: str,
    top_k: int,
    k: int,
) -> tuple:
    """Load suite, run queries, compute metrics.

    Returns:
        (EvalReport, list[dict] raw_results) or raises on error.
    """
    from rag.infra.indexes.index_manager import IndexManager
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.eval_pipeline import run_eval
    from rag.pipelines.query_pipeline import QueryPipeline

    suite_path = _SUITES[suite_name]
    entries = _load_suite(suite_path)

    trace_store = SQLiteTraceStore(db_path)
    manager = IndexManager(index_dir)
    pipeline = QueryPipeline(
        keyword_index=manager.bm25,
        vector_index=manager.faiss,
        trace_store=trace_store,
        top_k=top_k,
        answer_composer=None,
    )

    raw_results = []
    for entry in entries:
        query = entry.get("query", "")
        expected_sources = entry.get("expected_sources", [])
        expected_behavior = entry.get("expected_behavior", "")
        query_id = entry.get("query_id", query[:40])

        t0 = time.perf_counter()
        qr = pipeline.query(query)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_results.append(
            {
                "query_id": query_id,
                "query": query,
                "retrieved": [c.chunk_id for c in qr.candidates],
                "relevant": expected_sources,
                "candidates": qr.candidates,
                "expected_behavior": expected_behavior,
                "query_latency_ms": elapsed_ms,
                "error": qr.error,
            }
        )

    report = run_eval(raw_results, k=k)
    return report, raw_results


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("📊 Evaluation Panel")
st.caption("Run evaluation suites and review RAG quality metrics.")

tab_run, tab_results, tab_glossary = st.tabs(
    ["▶ Run Evaluation", "📈 Results", "📖 Metric Glossary"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Run Evaluation
# ---------------------------------------------------------------------------

with tab_run:
    st.subheader("Configuration")

    with st.form("eval_config_form"):
        col_l, col_r = st.columns(2)

        with col_l:
            suite_choice = st.selectbox(
                "Evaluation suite",
                options=list(_SUITES.keys()),
                format_func=lambda x: {
                    "example_queries": "📋 Example Queries (representative)",
                    "failure_cases": "⚠️ Failure Cases (challenging)",
                }[x],
            )
            top_k = st.number_input(
                "Top-K candidates to retrieve", min_value=1, max_value=50, value=10
            )

        with col_r:
            db_path = st.text_input("Database path", value=_DEFAULT_DB)
            index_dir = st.text_input("Index directory", value=_DEFAULT_INDEX)
            k_cutoff = st.number_input(
                "Metric cut-off (K)", min_value=1, max_value=50, value=10
            )

        submitted = st.form_submit_button(
            "▶ Run Evaluation",
            disabled=st.session_state.eval_running,
            use_container_width=True,
        )

    if submitted:
        db = Path(db_path)
        if not db.exists():
            st.error(f"Database not found: `{db_path}`. Run ingestion first.")
        else:
            st.session_state.eval_running = True
            st.session_state.eval_error = None
            st.session_state.eval_report = None

            with st.spinner(f"Running `{suite_choice}` suite…"):
                try:
                    report, _ = _run_evaluation(
                        suite_name=suite_choice,
                        db_path=db_path,
                        index_dir=index_dir,
                        top_k=int(top_k),
                        k=int(k_cutoff),
                    )
                    st.session_state.eval_report = report
                    st.session_state.eval_suite = suite_choice
                    st.session_state.eval_running = False
                    st.success(
                        f"Evaluation complete — {report.num_queries} queries "
                        f"evaluated.  Switch to the **Results** tab to review."
                    )
                except Exception as exc:
                    st.session_state.eval_running = False
                    st.session_state.eval_error = str(exc)
                    st.error(f"Evaluation failed: {exc}")

    # Suite preview
    with st.expander("Preview suite entries", expanded=False):
        preview_suite = st.selectbox(
            "Suite to preview",
            options=list(_SUITES.keys()),
            key="preview_suite_select",
        )
        suite_file = _SUITES.get(str(preview_suite))
        if suite_file is not None and suite_file.exists():
            entries = _load_suite(suite_file)
            st.write(f"**{len(entries)} entries** in `{suite_file.name}`")
            for i, e in enumerate(entries[:5]):
                with st.expander(
                    f"[{e.get('query_id', i+1)}] {e.get('query', '')[:80]}",
                    expanded=False,
                ):
                    st.json(e)
            if len(entries) > 5:
                st.caption(f"… and {len(entries) - 5} more entries.")
        else:
            st.warning(f"Suite file not found: `{suite_file}`")


# ---------------------------------------------------------------------------
# Tab 2 — Results
# ---------------------------------------------------------------------------

with tab_results:
    report = st.session_state.eval_report

    if report is None:
        st.info("No evaluation results yet. Run an evaluation in the **▶ Run Evaluation** tab.")
    else:
        from rag.app.studio.components.metrics_table import (
            render_efficiency_metrics,
            render_metrics_summary,
            render_per_case_table,
            render_source_attribution,
        )

        suite_label = {
            "example_queries": "📋 Example Queries",
            "failure_cases": "⚠️ Failure Cases",
        }.get(st.session_state.eval_suite, st.session_state.eval_suite)

        st.markdown(f"**Suite:** {suite_label} &nbsp;·&nbsp; **Queries:** {report.num_queries} &nbsp;·&nbsp; **K:** {report.k}")
        st.divider()

        render_metrics_summary(report)
        st.divider()

        render_source_attribution(report)
        st.divider()

        render_efficiency_metrics(report)
        st.divider()

        render_per_case_table(report.per_query, k=report.k)


# ---------------------------------------------------------------------------
# Tab 3 — Metric Glossary
# ---------------------------------------------------------------------------

with tab_glossary:
    from rag.app.studio.components.metric_glossary import GLOSSARY

    st.subheader("Metric Glossary")
    st.caption("Definitions, direction, and common pitfalls for every metric shown in this panel.")

    _SECTIONS = {
        "Retrieval Metrics": ["recall_at_k", "mrr", "ndcg_at_k"],
        "Source Attribution": ["bm25_only", "vector_only", "both"],
        "Efficiency Metrics": [
            "token_saved_est",
            "mean_query_latency_ms",
            "mean_ingest_latency_ms",
            "skipped_chunks",
            "changed_chunks",
        ],
    }

    _DIRECTION_LABEL = {
        "higher_is_better": "↑ Higher is better",
        "lower_is_better": "↓ Lower is better",
        "neutral": "→ Context-dependent",
    }

    for section_title, keys in _SECTIONS.items():
        st.markdown(f"### {section_title}")
        for key in keys:
            entry = GLOSSARY.get(key)
            if not entry:
                continue
            with st.expander(f"**{entry['label']}** — `{key}`", expanded=False):
                st.markdown(entry["description"])
                dir_text = _DIRECTION_LABEL.get(entry["direction"], entry["direction"])
                st.markdown(f"**{dir_text}**")
                st.warning(f"⚠️ {entry['pitfalls']}")
        st.divider()
