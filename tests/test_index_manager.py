"""Tests for IndexManager — index startup loader (Task 4.7)."""

import tempfile
from pathlib import Path

import pytest

from rag.core.contracts.chunk import Chunk
from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.infra.indexes.bm25_local import BM25LocalIndex
from rag.infra.indexes.faiss_local import FaissLocalIndex
from rag.infra.indexes.index_manager import IndexManager
from rag.infra.stores.docstore_sqlite import SQLiteDocStore, init_schema as init_doc_schema
from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore, init_schema as init_trace_schema
from rag.pipelines.ingest_pipeline import IngestPipeline

_DIM = 8


# ---------------------------------------------------------------------------
# Stub embedding provider
# ---------------------------------------------------------------------------


class StubEmbeddingProvider(BaseEmbeddingProvider):
    @property
    def dim(self) -> int:
        return _DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        return [[float(len(t) % _DIM == i) for i in range(_DIM)] for t in texts]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stores(db_path: str):
    init_doc_schema(db_path)
    init_trace_schema(db_path)
    return SQLiteDocStore(db_path), SQLiteTraceStore(db_path)


def _make_txt(tmp_path: Path) -> str:
    p = tmp_path / "sample.txt"
    p.write_text(
        "First paragraph about machine learning.\n\n"
        "Second paragraph about neural networks.\n\n"
        "Third paragraph about transformers and attention.\n",
        encoding="utf-8",
    )
    return str(p)


# ---------------------------------------------------------------------------
# First-run (no files on disk)
# ---------------------------------------------------------------------------


def test_fresh_start_creates_empty_indexes(tmp_path):
    """IndexManager with no saved files should give working empty indexes."""
    manager = IndexManager(tmp_path / "indexes")
    assert isinstance(manager.bm25, BM25LocalIndex)
    assert isinstance(manager.faiss, FaissLocalIndex)
    # Empty indexes return no results
    assert manager.bm25.search("query", top_k=5) == []
    assert manager.faiss.search([0.0] * _DIM, top_k=5) == []


# ---------------------------------------------------------------------------
# Save → reload cycle (simulates process restart)
# ---------------------------------------------------------------------------


def test_save_and_reload_restores_bm25(tmp_path):
    """BM25 state saved after ingest must be present after simulated restart."""
    index_dir = str(tmp_path / "indexes")
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    txt = _make_txt(tmp_path)

    # --- Process 1: ingest ---
    manager1 = IndexManager(index_dir)
    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=StubEmbeddingProvider(),
        keyword_index=manager1.bm25,
        vector_index=manager1.faiss,
    )
    result = pipeline.ingest(txt)
    assert result.error is None
    manager1.save()

    # --- Process 2: restart (fresh IndexManager pointing at same dir) ---
    manager2 = IndexManager(index_dir)
    candidates = manager2.bm25.search("machine learning", top_k=5)
    assert len(candidates) > 0


def test_save_and_reload_restores_faiss(tmp_path):
    """FAISS state saved after ingest must be present after simulated restart."""
    index_dir = str(tmp_path / "indexes")
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    txt = _make_txt(tmp_path)

    # --- Process 1: ingest ---
    manager1 = IndexManager(index_dir)
    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=StubEmbeddingProvider(),
        keyword_index=manager1.bm25,
        vector_index=manager1.faiss,
    )
    pipeline.ingest(txt)
    manager1.save()

    # --- Process 2: restart ---
    manager2 = IndexManager(index_dir)
    query = [0.0] * _DIM
    candidates = manager2.faiss.search(query, top_k=5)
    assert len(candidates) > 0


# ---------------------------------------------------------------------------
# Reload method
# ---------------------------------------------------------------------------


def test_reload_picks_up_new_data(tmp_path):
    """reload() must replace in-memory state with updated on-disk indexes."""
    index_dir = str(tmp_path / "indexes")
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    txt = _make_txt(tmp_path)

    manager = IndexManager(index_dir)

    # Ingest via separate index references, save
    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=StubEmbeddingProvider(),
        keyword_index=manager.bm25,
        vector_index=manager.faiss,
    )
    pipeline.ingest(txt)
    manager.save()

    # Simulate external update: reload from disk
    manager.reload()
    assert manager.bm25.search("transformers", top_k=5)


# ---------------------------------------------------------------------------
# Corrupt / partial file handling
# ---------------------------------------------------------------------------


def test_corrupted_bm25_file_starts_empty(tmp_path):
    """A corrupt bm25.pkl must not crash — IndexManager falls back to empty."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    (index_dir / "bm25.pkl").write_bytes(b"not valid pickle data")

    manager = IndexManager(str(index_dir))  # must not raise
    assert manager.bm25.search("anything", top_k=5) == []
