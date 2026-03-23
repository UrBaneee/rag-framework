"""Integration tests for QueryPipeline — Task 5.3."""

from pathlib import Path

import pytest

from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.infra.indexes.bm25_local import BM25LocalIndex
from rag.infra.indexes.faiss_local import FaissLocalIndex
from rag.infra.stores.docstore_sqlite import SQLiteDocStore, init_schema as init_doc_schema
from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore, init_schema as init_trace_schema
from rag.pipelines.ingest_pipeline import IngestPipeline
from rag.pipelines.query_pipeline import QueryPipeline, QueryResult

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
# Fixtures
# ---------------------------------------------------------------------------


def _make_stores(db_path: str):
    init_doc_schema(db_path)
    init_trace_schema(db_path)
    return SQLiteDocStore(db_path), SQLiteTraceStore(db_path)


@pytest.fixture()
def ingested_indexes(tmp_path: Path):
    """Ingest a sample text file and return ready-to-query indexes."""
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    provider = StubEmbeddingProvider()
    bm25 = BM25LocalIndex()
    faiss = FaissLocalIndex()

    # Write a sample document
    txt = tmp_path / "doc.txt"
    txt.write_text(
        "Retrieval augmented generation combines search with language models.\n\n"
        "BM25 is a keyword-based retrieval algorithm used in search engines.\n\n"
        "Vector search finds semantically similar documents using embeddings.\n\n"
        "Fusion combines multiple retrieval signals into a single ranked list.\n",
        encoding="utf-8",
    )

    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=provider,
        keyword_index=bm25,
        vector_index=faiss,
    )
    result = pipeline.ingest(str(txt))
    assert result.error is None, f"Ingest failed: {result.error}"

    return bm25, faiss, trace_store, provider


# ---------------------------------------------------------------------------
# Basic query tests
# ---------------------------------------------------------------------------


def test_query_returns_candidates(ingested_indexes, tmp_path):
    bm25, faiss, trace_store, provider = ingested_indexes
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
    )
    result = pipeline.query("retrieval augmented generation")
    assert result.error is None
    assert len(result.candidates) > 0


def test_query_returns_citations(ingested_indexes, tmp_path):
    bm25, faiss, trace_store, provider = ingested_indexes
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
    )
    result = pipeline.query("BM25 keyword search")
    assert result.error is None
    assert len(result.citations) > 0
    # Citations must be 1-based
    assert result.citations[0].ref_number == 1


def test_query_citations_match_candidates(ingested_indexes):
    bm25, faiss, trace_store, provider = ingested_indexes
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
    )
    result = pipeline.query("embeddings vector")
    assert len(result.citations) == len(result.candidates)
    for i, (cit, cand) in enumerate(zip(result.citations, result.candidates), start=1):
        assert cit.ref_number == i
        assert cit.chunk_id == cand.chunk_id


def test_query_records_run_in_trace_store(ingested_indexes):
    bm25, faiss, trace_store, provider = ingested_indexes
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
    )
    result = pipeline.query("fusion ranking")
    assert result.run_id
    runs = trace_store.list_runs(run_type="query")
    assert len(runs) >= 1


def test_query_elapsed_ms_populated(ingested_indexes):
    bm25, faiss, trace_store, provider = ingested_indexes
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
    )
    result = pipeline.query("search algorithm")
    assert result.elapsed_ms > 0


def test_query_top_k_limits_candidates(ingested_indexes):
    bm25, faiss, trace_store, provider = ingested_indexes
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
        top_k=2,
    )
    result = pipeline.query("language model")
    assert len(result.candidates) <= 2


# ---------------------------------------------------------------------------
# BM25-only mode (no vector index)
# ---------------------------------------------------------------------------


def test_query_bm25_only_mode(ingested_indexes):
    """Pipeline without vector index falls back to BM25-only retrieval."""
    bm25, _, trace_store, _ = ingested_indexes
    pipeline = QueryPipeline(
        keyword_index=bm25,
        trace_store=trace_store,
    )
    result = pipeline.query("retrieval")
    assert result.error is None
    assert len(result.candidates) > 0
