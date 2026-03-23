"""Trace viewer component — renders ingestion pipeline run events in Streamlit."""

from __future__ import annotations

from typing import Any

import streamlit as st

# ---------------------------------------------------------------------------
# Stage ordering and display config
# ---------------------------------------------------------------------------

# Human-readable labels for known run_type values in order of pipeline stages
_STAGE_ORDER: list[str] = [
    "ingest_start",
    "sniff",
    "parse",
    "quality_gate",
    "clean",
    "block_split",
    "chunk_pack",
    "embed",
    "ingest_complete",
    "ingest_error",
]

_STAGE_LABELS: dict[str, str] = {
    "ingest_start":    "🚀 Ingest Start",
    "sniff":           "🔍 Sniffer",
    "parse":           "📄 Parser",
    "quality_gate":    "🚦 Quality Gate",
    "clean":           "🧹 Cleaner",
    "block_split":     "✂️  Block Splitter",
    "chunk_pack":      "📦 Chunk Packer",
    "embed":           "🔢 Embedder",
    "ingest_complete": "✅ Ingest Complete",
    "ingest_error":    "❌ Ingest Error",
}

_STAGE_COLORS: dict[str, str] = {
    "ingest_complete": "🟢",
    "ingest_error":    "🔴",
    "quality_gate":    "🟡",
}


def render_run_selector(runs: list[dict[str, Any]], label: str = "Select a run") -> str | None:
    """Render a selectbox for choosing a run_id from a list of runs.

    Args:
        runs: List of run dicts as returned by ``TraceStore.list_runs()``.
        label: Widget label string.

    Returns:
        The selected ``run_id``, or None if no runs are available.
    """
    if not runs:
        st.info("No ingestion runs found. Upload a document on the Ingestion Manager page first.")
        return None

    options = {
        f"{r['run_id'][:8]}… | {r.get('created_at', '')[:19]} | "
        f"{r.get('metadata', {}).get('source_path', 'unknown')!r}": r["run_id"]
        for r in runs
    }
    choice = st.selectbox(label, list(options.keys()))
    return options[choice] if choice else None


def render_run_events(events: list[dict[str, Any]]) -> None:
    """Render all pipeline events for a single ingestion run.

    Events are grouped by ``run_type`` and displayed in pipeline-stage order.
    Unknown run_types are shown at the bottom under "Other events".

    Args:
        events: List of run event dicts with keys ``run_type``, ``metadata``,
                ``created_at``, ``run_id``.
    """
    if not events:
        st.warning("No events found for this run.")
        return

    # Group events by run_type
    by_type: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        rt = ev.get("run_type", "unknown")
        by_type.setdefault(rt, []).append(ev)

    # Render known stages in order
    for stage in _STAGE_ORDER:
        if stage not in by_type:
            continue
        label = _STAGE_LABELS.get(stage, stage)
        color = _STAGE_COLORS.get(stage, "⚪")
        with st.expander(f"{color} {label}", expanded=(stage in ("ingest_complete", "ingest_error"))):
            for ev in by_type[stage]:
                _render_event_body(ev)

    # Render any unknown run_types
    known = set(_STAGE_ORDER)
    extras = {rt: evs for rt, evs in by_type.items() if rt not in known}
    if extras:
        with st.expander("📋 Other events", expanded=False):
            for rt, evs in sorted(extras.items()):
                st.markdown(f"**{rt}**")
                for ev in evs:
                    _render_event_body(ev)


def _render_event_body(event: dict[str, Any]) -> None:
    """Render a single event's metadata payload."""
    meta = event.get("metadata", {})
    created_at = event.get("created_at", "")
    if created_at:
        st.caption(f"Recorded at: {created_at[:19]}")

    if not meta:
        st.markdown("*(no metadata)*")
        return

    # Surface key scalar fields as metrics where useful
    metric_keys = [
        ("block_count", "Blocks"),
        ("chunk_count", "Chunks"),
        ("elapsed_ms", "Elapsed ms"),
        ("embed_tokens", "Embed tokens"),
        ("char_count", "Characters"),
    ]
    metric_values = [(label, meta[key]) for key, label in metric_keys if key in meta]
    if metric_values:
        cols = st.columns(min(len(metric_values), 4))
        for col, (label, val) in zip(cols, metric_values):
            col.metric(label, f"{val:.0f}" if isinstance(val, float) else val)

    # Show all fields as JSON for full detail
    st.json(meta, expanded=False)


def render_run_summary_table(runs: list[dict[str, Any]]) -> None:
    """Render a compact summary table of recent runs.

    Args:
        runs: List of run dicts (most recent first).
    """
    if not runs:
        return

    import pandas as pd

    rows = []
    for r in runs:
        meta = r.get("metadata", {})
        rows.append({
            "Run ID": r["run_id"][:12] + "…",
            "Type": r.get("run_type", ""),
            "Source": meta.get("source_path", meta.get("query", ""))[-40:] if (
                meta.get("source_path") or meta.get("query")
            ) else "—",
            "Elapsed ms": meta.get("elapsed_ms", "—"),
            "Chunks": meta.get("chunk_count", "—"),
            "Time": r.get("created_at", "")[:19],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
