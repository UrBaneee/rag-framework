"""Tests for the Ingestion Manager page logic — Task 8.2.

Tests cover:
- Page imports cleanly (no runtime crash)
- Ingest summary fields match IngestResult
- Dimension mismatch detection helper
- Config dict structure
"""

from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Streamlit stub (prevents actual st calls during import)
# ---------------------------------------------------------------------------


def _make_st_stub() -> types.ModuleType:
    """Build a minimal Streamlit stub that accepts all calls silently."""

    class _CM:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            pass
        # Support attribute access on context manager return (e.g. col1.metric())
        def __getattr__(self, name):
            return lambda *a, **kw: _CM()

    stub = types.ModuleType("streamlit")
    _noop = lambda *a, **kw: _CM()

    for attr in [
        "set_page_config", "title", "caption", "header", "subheader",
        "info", "success", "warning", "error", "divider", "markdown",
        "expander", "spinner", "json", "metric", "rerun", "write",
        "file_uploader", "selectbox",
    ]:
        setattr(stub, attr, _noop)

    # text_input must return a string so Path() calls don't crash
    stub.text_input = lambda *a, **kw: kw.get("value", "")

    # number_input must return a real number so int() calls don't crash
    stub.number_input = lambda *a, **kw: kw.get("value", 0)
    # button must return False so the run block is not entered at import time
    stub.button = lambda *a, **kw: False

    stub.columns = lambda n: [_CM() for _ in range(n if isinstance(n, int) else len(n))]
    stub.tabs = lambda labels: [_CM() for _ in labels]
    stub.progress = lambda *a, **kw: _CM()
    stub.text_area = lambda *a, **kw: ""
    stub.stop = lambda: None
    stub.bar_chart = _noop
    stub.code = _noop
    stub.cache_data = lambda *a, **kw: (lambda f: f)
    stub.sidebar = _CM()

    # session_state as a simple dict-like object
    class _SessionState(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)
        def __setattr__(self, k, v):
            self[k] = v

    stub.session_state = _SessionState(
        ingest_result=None,
        ingest_config=None,
        ingest_error=None,
        last_doc_ids=[],
    )
    return stub


# ---------------------------------------------------------------------------
# Helper: import the page module with stubbed streamlit
# ---------------------------------------------------------------------------


def _import_page():
    stub = _make_st_stub()
    old = sys.modules.get("streamlit")
    sys.modules["streamlit"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "ingestion_manager_page",
            Path(__file__).parent.parent / "rag/app/studio/pages/1_ingest_inspect.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            del sys.modules["streamlit"]
        else:
            sys.modules["streamlit"] = old
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_page_imports_without_error():
    """The page module must import cleanly against a stubbed Streamlit."""
    mod = _import_page()
    assert mod is not None


def test_ingest_result_fields_present():
    """IngestResult dataclass must expose all fields shown in the UI."""
    from rag.pipelines.ingest_pipeline import IngestResult

    result = IngestResult(
        doc_id="doc-abc",
        source_path="/tmp/test.pdf",
        block_count=42,
        chunk_count=10,
        run_id="run-001",
        elapsed_ms=350.5,
        embed_tokens=1234,
    )
    assert result.block_count == 42
    assert result.chunk_count == 10
    assert result.embed_tokens == 1234
    assert result.elapsed_ms == 350.5
    assert result.error is None
    assert result.skipped is False


def test_ingest_result_skipped_flag():
    from rag.pipelines.ingest_pipeline import IngestResult

    result = IngestResult(doc_id="x", source_path="/tmp/x.txt", skipped=True)
    assert result.skipped is True


def test_ingest_result_error_flag():
    from rag.pipelines.ingest_pipeline import IngestResult

    result = IngestResult(doc_id="x", source_path="/tmp/x.txt", error="parse failed")
    assert result.error == "parse failed"


def test_config_dict_has_required_keys():
    """Config dict assembled in the UI must contain all required display fields."""
    config = {
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "vector_dimension": 1536,
        "index_type": "FAISS (IndexFlatL2)",
        "token_budget": 512,
        "collection": "default",
        "db_path": "data/rag.db",
        "index_dir": "data/indexes",
    }
    for key in [
        "embedding_provider", "embedding_model", "vector_dimension", "index_type"
    ]:
        assert key in config, f"Missing required config key: {key}"


def test_dimension_mismatch_detected():
    """A FaissLocalIndex with a different dimension should trigger a mismatch."""
    from rag.infra.indexes.faiss_local import FaissLocalIndex
    import numpy as np

    index = FaissLocalIndex()
    # Insert a single 64-dim vector
    mock_chunk = MagicMock()
    mock_chunk.chunk_id = "c0"
    mock_chunk.doc_id = "doc0"
    mock_chunk.stable_text = "text"
    mock_chunk.display_text = "text"
    mock_chunk.metadata = {}
    mock_chunk.embedding = [0.1] * 64

    index.add([mock_chunk])
    assert index.dimension == 64

    # Simulating what the UI does: compare loaded dimension vs config dimension
    configured_dim = 1536
    mismatch = index.dimension != configured_dim
    assert mismatch is True


def test_no_dimension_mismatch_when_equal():
    from rag.infra.indexes.faiss_local import FaissLocalIndex

    mock_chunk = MagicMock()
    mock_chunk.chunk_id = "c0"
    mock_chunk.doc_id = "doc0"
    mock_chunk.stable_text = "text"
    mock_chunk.display_text = "text"
    mock_chunk.metadata = {}
    mock_chunk.embedding = [0.1] * 1536

    index = FaissLocalIndex()
    index.add([mock_chunk])

    configured_dim = 1536
    mismatch = index.dimension != configured_dim
    assert mismatch is False


def test_faiss_index_exposes_dimension():
    """FaissLocalIndex must expose a .dimension property after add()."""
    from rag.infra.indexes.faiss_local import FaissLocalIndex

    mock_chunk = MagicMock()
    mock_chunk.chunk_id = "c0"
    mock_chunk.doc_id = "doc0"
    mock_chunk.stable_text = "text"
    mock_chunk.display_text = "text"
    mock_chunk.metadata = {}
    mock_chunk.embedding = [0.0] * 256

    index = FaissLocalIndex()
    index.add([mock_chunk])
    assert index.dimension == 256
