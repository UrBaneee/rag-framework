"""Tests for query traces page and candidate_table component — Task 8.4."""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock


BASE = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Streamlit stub factory
# ---------------------------------------------------------------------------

def _make_st_stub(submitted: bool = False, query_text: str = "") -> types.ModuleType:
    class _CM:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __getattr__(self, name): return lambda *a, **kw: _CM()

    stub = types.ModuleType("streamlit")
    _noop = lambda *a, **kw: _CM()

    for attr in [
        "set_page_config", "title", "caption", "header", "subheader",
        "info", "success", "warning", "error", "divider", "markdown",
        "expander", "spinner", "metric", "rerun", "write", "dataframe",
        "json", "selectbox", "checkbox",
    ]:
        setattr(stub, attr, _noop)

    stub.text_input = lambda *a, **kw: kw.get("value", "")
    stub.text_area = lambda *a, **kw: query_text
    stub.number_input = lambda *a, **kw: kw.get("value", 0)
    stub.button = lambda *a, **kw: False
    stub.columns = lambda n: [_CM() for _ in range(n if isinstance(n, int) else len(n))]
    stub.sidebar = _CM()

    # form_submit_button is called both as st.form_submit_button and inside
    # the form context manager — stub both paths
    stub.form_submit_button = lambda *a, **kw: submitted

    class _Form(_CM):
        def form_submit_button(self, *a, **kw): return submitted

    stub.form = lambda *a, **kw: _Form()
    stub.cache_data = lambda *a, **kw: (lambda f: f)

    class _SessionState(dict):
        def __getattr__(self, k):
            try: return self[k]
            except KeyError: raise AttributeError(k)
        def __setattr__(self, k, v): self[k] = v

    stub.session_state = _SessionState(
        query_result=None,
        query_running=False,
        query_error=None,
        last_query="",
    )
    return stub


def _import_with_st(file_path: str, module_name: str, stub=None):
    if stub is None:
        stub = _make_st_stub()
    old = sys.modules.get("streamlit")
    sys.modules["streamlit"] = stub
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = old
    return mod


# ---------------------------------------------------------------------------
# candidate_table component tests
# ---------------------------------------------------------------------------

def test_candidate_table_imports():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table",
    )
    assert mod is not None


def test_render_candidate_table_no_candidates():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table2",
    )
    mod.render_candidate_table([])  # must not raise


def test_render_candidate_table_with_data():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table3",
    )
    cands = [
        {
            "chunk_id": "c001",
            "doc_id": "doc.pdf",
            "display_text": "RAG combines retrieval with generation.",
            "stable_text": "RAG combines retrieval with generation.",
            "bm25_score": 0.8,
            "vector_score": 0.9,
            "rrf_score": 0.045,
            "rerank_score": None,
            "final_score": 0.045,
            "retrieval_source": "hybrid",
            "metadata": {"start_page": 3},
        }
    ]
    mod.render_candidate_table(cands)  # must not raise


def test_render_candidate_table_with_rerank_change():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table4",
    )
    cands = [
        {"chunk_id": "c001", "doc_id": "doc.pdf", "display_text": "A",
         "stable_text": "A", "bm25_score": 0.8, "vector_score": None,
         "rrf_score": 0.04, "rerank_score": 0.95, "final_score": 0.95,
         "retrieval_source": "bm25", "metadata": {}},
        {"chunk_id": "c002", "doc_id": "doc.pdf", "display_text": "B",
         "stable_text": "B", "bm25_score": 0.9, "vector_score": None,
         "rrf_score": 0.05, "rerank_score": 0.60, "final_score": 0.60,
         "retrieval_source": "bm25", "metadata": {}},
    ]
    # c002 was rank 1 before rerank, now rank 2 → should show ▼1
    pre_rerank = ["c002", "c001"]
    mod.render_candidate_table(cands, pre_rerank_ids=pre_rerank)  # must not raise


def test_render_context_packing_details():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table5",
    )
    packed = [{"chunk_id": "c001", "doc_id": "doc.pdf", "display_text": "A",
               "stable_text": "A", "metadata": {"token_count": 50}}]
    all_cands = packed + [{"chunk_id": "c002", "doc_id": "doc.pdf",
                           "display_text": "B", "stable_text": "B", "metadata": {}}]
    mod.render_context_packing_details(
        packed_candidates=packed,
        all_candidates=all_cands,
        context_top_k=3,
        token_budget=512,
        packed_tokens=50,
        truncated=False,
    )  # must not raise


def test_render_context_packing_shows_truncated():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table6",
    )
    # With truncated=True — the component should set metric to "Yes ⚠️"
    mod.render_context_packing_details(
        packed_candidates=[],
        all_candidates=[{"chunk_id": "c1", "doc_id": "d", "display_text": "T",
                         "stable_text": "T", "metadata": {}}],
        context_top_k=3,
        token_budget=100,
        packed_tokens=0,
        truncated=True,
    )


def test_render_answer_section():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table7",
    )
    mod.render_answer_section(
        answer_text="RAG is retrieval-augmented generation.[1]",
        citations=[{"ref_number": 1, "chunk_id": "c001", "source_label": "doc.pdf — page 1"}],
        abstained=False,
        prompt_tokens=40,
        completion_tokens=15,
        total_tokens=55,
        generation_latency_ms=120.0,
    )


def test_render_answer_section_abstained():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table8",
    )
    mod.render_answer_section(
        answer_text="I don't have enough information.",
        citations=[],
        abstained=True,
        prompt_tokens=30,
        completion_tokens=8,
        total_tokens=38,
        generation_latency_ms=80.0,
    )


def test_score_bar_helper():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table9",
    )
    bar = mod._score_bar(0.5, max_val=1.0)
    assert "█" in bar
    assert "0.5" in bar


def test_score_bar_none():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/components/candidate_table.py"),
        "candidate_table10",
    )
    assert mod._score_bar(None) == "—"


# ---------------------------------------------------------------------------
# Query traces page tests
# ---------------------------------------------------------------------------

def test_query_traces_page_imports():
    mod = _import_with_st(
        str(BASE / "rag/app/studio/pages/3_query_traces.py"),
        "query_traces_page",
    )
    assert mod is not None


def test_query_traces_page_empty_query_not_executed():
    """When submitted=True but query_text='', no pipeline call should occur."""
    stub = _make_st_stub(submitted=True, query_text="  ")
    # Track if rerun was called — it should NOT be called when query is empty
    rerun_called = []
    stub.rerun = lambda: rerun_called.append(True)
    _import_with_st(
        str(BASE / "rag/app/studio/pages/3_query_traces.py"),
        "query_traces_page_empty",
        stub=stub,
    )
    # Pipeline should not have been triggered — no IndexManager import attempted
    # The page shows an error and does NOT call rerun
    assert len(rerun_called) == 0


def test_query_traces_page_no_submit_no_run():
    """When submitted=False, pipeline is never invoked."""
    stub = _make_st_stub(submitted=False, query_text="What is RAG?")
    rerun_called = []
    stub.rerun = lambda: rerun_called.append(True)
    _import_with_st(
        str(BASE / "rag/app/studio/pages/3_query_traces.py"),
        "query_traces_no_submit",
        stub=stub,
    )
    assert len(rerun_called) == 0
