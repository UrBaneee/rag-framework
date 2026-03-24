"""Metrics table component — renders evaluation metrics in Streamlit.

All metric labels and tooltips come from ``metric_glossary.py`` to
ensure consistency across the UI.

Public functions:
- ``render_metrics_summary(report)``      — aggregate metric tiles
- ``render_source_attribution(report)``  — BM25/vector/both ratios
- ``render_efficiency_metrics(report)``  — latency and token savings
- ``render_per_case_table(results)``     — per-query outcome rows
- ``render_answer_quality(ragas_metrics)`` — RAGAS answer quality tiles
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from rag.app.studio.components.metric_glossary import label, tooltip


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


def render_metrics_summary(report: Any) -> None:
    """Render aggregate retrieval metrics as a row of st.metric tiles.

    Args:
        report: ``EvalReport`` instance from ``run_eval()``.
    """
    st.subheader("Retrieval Metrics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label=label("recall_at_k") + f"@{report.k}",
            value=f"{report.mean_recall_at_k:.3f}",
            help=tooltip("recall_at_k"),
        )
    with c2:
        st.metric(
            label=label("mrr"),
            value=f"{report.mrr:.3f}",
            help=tooltip("mrr"),
        )
    with c3:
        st.metric(
            label=label("ndcg_at_k") + f"@{report.k}",
            value=f"{report.mean_ndcg_at_k:.3f}",
            help=tooltip("ndcg_at_k"),
        )


# ---------------------------------------------------------------------------
# Source attribution
# ---------------------------------------------------------------------------


def render_source_attribution(report: Any) -> None:
    """Render source attribution ratios (BM25 / vector / both).

    Args:
        report: ``EvalReport`` instance from ``run_eval()``.
    """
    sa = report.source_attribution
    st.subheader("Source Attribution")

    if sa.total_candidates == 0:
        st.info("No candidate data available — run with `candidates` populated.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            label=label("bm25_only"),
            value=f"{sa.bm25_only:.1%}",
            help=tooltip("bm25_only"),
        )
    with c2:
        st.metric(
            label=label("vector_only"),
            value=f"{sa.vector_only:.1%}",
            help=tooltip("vector_only"),
        )
    with c3:
        st.metric(
            label=label("both"),
            value=f"{sa.both:.1%}",
            help=tooltip("both"),
        )
    with c4:
        st.metric(
            label="Total candidates",
            value=str(sa.total_candidates),
        )


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------


def render_efficiency_metrics(report: Any) -> None:
    """Render system efficiency metrics.

    Args:
        report: ``EvalReport`` instance from ``run_eval()``.
    """
    eff = report.efficiency
    st.subheader("Efficiency Metrics")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        val = f"{eff.mean_query_latency_ms:.1f} ms" if eff.mean_query_latency_ms is not None else "—"
        st.metric(
            label=label("mean_query_latency_ms"),
            value=val,
            help=tooltip("mean_query_latency_ms"),
        )
    with c2:
        val = f"{eff.mean_ingest_latency_ms:.1f} ms" if eff.mean_ingest_latency_ms is not None else "—"
        st.metric(
            label=label("mean_ingest_latency_ms"),
            value=val,
            help=tooltip("mean_ingest_latency_ms"),
        )
    with c3:
        val = f"{eff.token_saved_est:.0f}" if eff.token_saved_est is not None else "—"
        st.metric(
            label=label("token_saved_est"),
            value=val,
            help=tooltip("token_saved_est"),
        )
    with c4:
        # N/A until Task 11.2 block-diff is complete
        val = str(eff.skipped_chunks) if eff.skipped_chunks is not None else "N/A"
        st.metric(
            label=label("skipped_chunks"),
            value=val,
            help=tooltip("skipped_chunks"),
        )
    with c5:
        val = str(eff.changed_chunks) if eff.changed_chunks is not None else "N/A"
        st.metric(
            label=label("changed_chunks"),
            value=val,
            help=tooltip("changed_chunks"),
        )


# ---------------------------------------------------------------------------
# Per-case results table
# ---------------------------------------------------------------------------

_OUTCOME_ICON = {
    "PASS": "✅",
    "FAIL": "❌",
    "SKIP": "⚪",
}

_BEHAVIOR_ICON = {
    "answer": "💬",
    "abstain": "🚫",
    "warn": "⚠️",
}


def _case_outcome(q: Any, k: int) -> str:
    """Determine PASS / FAIL / SKIP for a QueryEvalResult."""
    if q.relevant_count == 0:
        return "SKIP"
    top_k_retrieved = q.retrieved_count > 0 and q.recall_at_k > 0
    return "PASS" if top_k_retrieved else "FAIL"


def render_per_case_table(per_query: list[Any], k: int) -> None:
    """Render a per-query results table with outcome, expected vs actual behavior.

    Args:
        per_query: List of ``QueryEvalResult`` objects from ``EvalReport.per_query``.
        k: Cut-off depth used for evaluation.
    """
    st.subheader("Per-Case Results")

    if not per_query:
        st.info("No results to display.")
        return

    rows = []
    for q in per_query:
        outcome = _case_outcome(q, k)
        rows.append(
            {
                "Outcome": _OUTCOME_ICON.get(outcome, outcome) + " " + outcome,
                "Query ID": q.query_id or "—",
                "Expected": _BEHAVIOR_ICON.get(q.expected_behavior, "") + " " + (q.expected_behavior or "—"),
                "Actual": _BEHAVIOR_ICON.get(q.actual_behavior, "") + " " + (q.actual_behavior or "—"),
                f"Recall@{k}": f"{q.recall_at_k:.3f}",
                "MRR": f"{q.mrr:.3f}",
                f"nDCG@{k}": f"{q.ndcg_at_k:.3f}",
                "Retrieved": q.retrieved_count,
                "Relevant": q.relevant_count,
            }
        )

    # Summary counters
    pass_count = sum(1 for q in per_query if _case_outcome(q, k) == "PASS")
    fail_count = sum(1 for q in per_query if _case_outcome(q, k) == "FAIL")
    skip_count = sum(1 for q in per_query if _case_outcome(q, k) == "SKIP")

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("✅ Passed", pass_count)
    mc2.metric("❌ Failed", fail_count)
    mc3.metric("⚪ Skipped", skip_count)

    st.divider()
    st.dataframe(rows, use_container_width=True)


# ---------------------------------------------------------------------------
# Answer quality (RAGAS)
# ---------------------------------------------------------------------------


def render_answer_quality(ragas_metrics: Any) -> None:
    """Render RAGAS answer quality metrics as a row of st.metric tiles.

    Shows faithfulness, answer relevancy, and context precision.  If
    ``ragas_metrics`` is None or ``ragas_available`` is False, displays
    an informational message instead.

    Args:
        ragas_metrics: ``AnswerQualityMetrics`` instance from
            ``run_golden_eval()``, or ``None`` if not computed.
    """
    st.subheader("Answer Quality (RAGAS)")

    if ragas_metrics is None:
        st.info(
            "RAGAS answer quality metrics were not computed for this run. "
            "Use `--answer-quality` via the CLI or call `run_golden_eval()` "
            "to enable them."
        )
        return

    if not ragas_metrics.ragas_available:
        st.warning(
            "RAGAS is not installed.  Install it with:\n\n"
            "```\npip install ragas\n```"
        )
        return

    if ragas_metrics.num_evaluated == 0:
        st.info("RAGAS evaluation ran but no queries were successfully scored.")
        return

    st.caption(f"Evaluated {ragas_metrics.num_evaluated} queries against the golden answer set.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            label=label("faithfulness"),
            value=f"{ragas_metrics.mean_faithfulness:.3f}",
            help=tooltip("faithfulness"),
        )
    with c2:
        st.metric(
            label=label("answer_relevancy"),
            value=f"{ragas_metrics.mean_answer_relevancy:.3f}",
            help=tooltip("answer_relevancy"),
        )
    with c3:
        st.metric(
            label=label("context_precision"),
            value=f"{ragas_metrics.mean_context_precision:.3f}",
            help=tooltip("context_precision"),
        )

    if ragas_metrics.per_query:
        with st.expander("Per-query RAGAS scores", expanded=False):
            st.dataframe(ragas_metrics.per_query, use_container_width=True)
