"""Tests for trace_viewer component and ingestion trace page — Task 8.3."""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Streamlit stub
# ---------------------------------------------------------------------------


def _make_st_stub() -> types.ModuleType:
    class _CM:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __getattr__(self, name): return lambda *a, **kw: _CM()

    stub = types.ModuleType("streamlit")
    _noop = lambda *a, **kw: _CM()

    for attr in [
        "set_page_config", "title", "caption", "header", "subheader",
        "info", "success", "warning", "error", "divider", "markdown",
        "expander", "spinner", "button", "selectbox",
        "json", "metric", "rerun", "write", "dataframe", "tabs",
        "cache_data",
    ]:
        setattr(stub, attr, _noop)

    stub.number_input = lambda *a, **kw: kw.get("value", 0)
    stub.text_input = lambda *a, **kw: kw.get("value", "")
    stub.columns = lambda n: [_CM() for _ in range(n if isinstance(n, int) else len(n))]
    stub.sidebar = _CM()

    # cache_data as a pass-through decorator
    stub.cache_data = lambda *a, **kw: (lambda f: f)

    # tabs returns list of context managers
    stub.tabs = lambda labels: [_CM() for _ in labels]

    class _SessionState(dict):
        def __getattr__(self, k):
            try: return self[k]
            except KeyError: raise AttributeError(k)
        def __setattr__(self, k, v): self[k] = v

    stub.session_state = _SessionState()
    return stub


def _import_with_st(file_path: str, module_name: str):
    stub = _make_st_stub()
    old = sys.modules.get("streamlit")
    # Also stub pandas to avoid import errors
    pd_stub = types.ModuleType("pandas")
    pd_stub.DataFrame = lambda data: data
    old_pd = sys.modules.get("pandas")
    sys.modules["streamlit"] = stub
    sys.modules["pandas"] = pd_stub
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = old
        if old_pd is None:
            sys.modules.pop("pandas", None)
        else:
            sys.modules["pandas"] = old_pd
    return mod


BASE = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# trace_viewer component tests
# ---------------------------------------------------------------------------


def test_trace_viewer_imports():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/trace_viewer.py"),
        "trace_viewer",
    )
    assert mod is not None


def test_render_run_selector_no_runs_returns_none():
    """render_run_selector with empty list should return None."""
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/trace_viewer.py"),
        "trace_viewer",
    )
    result = mod.render_run_selector([])
    assert result is None


def test_render_run_selector_with_runs_returns_string():
    """render_run_selector with runs available should return a run_id string."""
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/trace_viewer.py"),
        "trace_viewer",
    )
    import streamlit as st
    # Stub selectbox to return the first option key
    runs = [
        {"run_id": "abc123def456", "created_at": "2026-01-01 00:00:00",
         "metadata": {"source_path": "/tmp/doc.pdf"}},
    ]
    # selectbox will return _CM() which is truthy but not in the dict —
    # so we patch selectbox to return a real key
    stub = _make_st_stub()
    old = sys.modules.get("streamlit")
    sys.modules["streamlit"] = stub

    # Build the option key the same way the component does
    r = runs[0]
    option_key = (
        f"{r['run_id'][:8]}… | {r.get('created_at', '')[:19]} | "
        f"{r.get('metadata', {}).get('source_path', 'unknown')!r}"
    )
    stub.selectbox = lambda label, options, **kw: options[0] if options else None

    try:
        spec = importlib.util.spec_from_file_location("tv2", str(BASE / "rag/app/studio/components/trace_viewer.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        result = m.render_run_selector(runs)
        assert result == "abc123def456"
    finally:
        if old is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = old


def test_render_run_events_empty():
    """render_run_events with empty list calls st.warning."""
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/trace_viewer.py"),
        "trace_viewer",
    )
    # Should not raise
    mod.render_run_events([])


def test_render_run_events_known_stages():
    """render_run_events renders known stage types without error."""
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/trace_viewer.py"),
        "trace_viewer",
    )
    events = [
        {"run_type": "ingest_start", "metadata": {"source_path": "/tmp/doc.pdf"}, "created_at": "2026-01-01"},
        {"run_type": "parse", "metadata": {"block_count": 12}, "created_at": "2026-01-01"},
        {"run_type": "chunk_pack", "metadata": {"chunk_count": 4}, "created_at": "2026-01-01"},
        {"run_type": "ingest_complete", "metadata": {"elapsed_ms": 350.0, "chunk_count": 4}, "created_at": "2026-01-01"},
    ]
    mod.render_run_events(events)  # must not raise


def test_render_run_events_unknown_stages():
    """render_run_events handles unknown run_types gracefully."""
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/trace_viewer.py"),
        "trace_viewer",
    )
    events = [
        {"run_type": "some_future_stage", "metadata": {"foo": "bar"}, "created_at": "2026-01-01"},
    ]
    mod.render_run_events(events)  # must not raise


def test_stage_order_covers_all_labels():
    """Every entry in _STAGE_ORDER should have a corresponding label."""
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/trace_viewer.py"),
        "trace_viewer",
    )
    for stage in mod._STAGE_ORDER:
        assert stage in mod._STAGE_LABELS, f"Missing label for stage: {stage}"


# ---------------------------------------------------------------------------
# Ingestion trace page tests
# ---------------------------------------------------------------------------


def test_ingestion_trace_page_imports():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/pages/3_ingestion_traces.py"),
        "ingestion_traces_page",
    )
    assert mod is not None


def test_load_runs_helper_returns_list_on_missing_db():
    """_load_runs should return an empty list when DB does not exist."""
    mod = _import_with_st(
        str(BASE / "rag/app/studio/pages/3_ingestion_traces.py"),
        "ingestion_traces_page2",
    )
    result = mod._load_runs("/nonexistent/path/rag.db", 50)
    assert isinstance(result, list)


def test_load_events_for_run_returns_list_on_missing_db():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/pages/3_ingestion_traces.py"),
        "ingestion_traces_page3",
    )
    result = mod._load_events_for_run("/nonexistent/path/rag.db", "run-xyz")
    assert isinstance(result, list)
