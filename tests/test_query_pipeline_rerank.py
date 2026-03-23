"""Tests for rerank stage in QueryPipeline — Task 6.3."""

from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.core.interfaces.reranker import BaseReranker
from rag.infra.indexes.bm25_local import BM25LocalIndex
from rag.infra.indexes.faiss_local import FaissLocalIndex
from rag.infra.rerank.noop import NoOpReranker
from rag.infra.stores.docstore_sqlite import SQLiteDocStore, init_schema as init_doc_schema
from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore, init_schema as init_trace_schema
from rag.pipelines.ingest_pipeline import IngestPipeline
from rag.pipelines.query_pipeline import QueryPipeline

_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubEmbeddingProvider(BaseEmbeddingProvider):
    @property
    def dim(self):
        return _DIM

    def embed(self, texts):
        return [[float(len(t) % _DIM == i) for i in range(_DIM)] for t in texts]


def _make_stores(db_path: str):
    init_doc_schema(db_path)
    init_trace_schema(db_path)
    return SQLiteDocStore(db_path), SQLiteTraceStore(db_path)


@pytest.fixture()
def ingested_env(tmp_path: Path):
    db = str(tmp_path / "rag.db")
    doc_store, trace_store = _make_stores(db)
    bm25 = BM25LocalIndex()
    faiss = FaissLocalIndex()
    provider = StubEmbeddingProvider()

    txt = tmp_path / "doc.txt"
    txt.write_text(
        "Reranking improves the precision of search results.\n\n"
        "BM25 is a keyword-based retrieval algorithm.\n\n"
        "Vector search finds semantically similar documents.\n\n"
        "Fusion combines multiple retrieval signals.\n",
        encoding="utf-8",
    )
    IngestPipeline(
        doc_store, trace_store,
        embedding_provider=provider,
        keyword_index=bm25,
        vector_index=faiss,
    ).ingest(str(txt))

    return bm25, faiss, trace_store, provider


# ---------------------------------------------------------------------------
# Rerank stage tests
# ---------------------------------------------------------------------------


def test_noop_reranker_integrated(ingested_env):
    """Pipeline with NoOpReranker returns results without error."""
    bm25, faiss, trace_store, provider = ingested_env
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
        reranker=NoOpReranker(),
    )
    result = pipeline.query("reranking retrieval")
    assert result.error is None
    assert len(result.candidates) > 0


def test_reranker_final_score_set(ingested_env):
    """final_score must be populated on all candidates after reranking."""
    bm25, faiss, trace_store, provider = ingested_env
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
        reranker=NoOpReranker(),
    )
    result = pipeline.query("search algorithm")
    for cand in result.candidates:
        assert cand.final_score is not None
        assert cand.rerank_score is not None


def test_pre_rerank_trace_recorded(ingested_env):
    """TraceStore must contain a query_pre_rerank event."""
    bm25, faiss, trace_store, provider = ingested_env
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
        reranker=NoOpReranker(),
    )
    pipeline.query("fusion signals")
    runs = trace_store.list_runs(run_type="query_pre_rerank")
    assert len(runs) >= 1


def test_post_rerank_trace_recorded(ingested_env):
    """TraceStore must contain a query_post_rerank event when reranker is set."""
    bm25, faiss, trace_store, provider = ingested_env
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
        reranker=NoOpReranker(),
    )
    pipeline.query("vector semantic search")
    runs = trace_store.list_runs(run_type="query_post_rerank")
    assert len(runs) >= 1


def test_no_post_rerank_trace_without_reranker(ingested_env):
    """Without a reranker, query_post_rerank must NOT be recorded."""
    bm25, faiss, trace_store, provider = ingested_env
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
        # no reranker
    )
    pipeline.query("keyword search")
    runs = trace_store.list_runs(run_type="query_post_rerank")
    assert len(runs) == 0


def test_custom_reranker_scores_applied(ingested_env):
    """A custom reranker's scores must appear on the returned candidates."""
    bm25, faiss, trace_store, provider = ingested_env

    class ScoringReranker(BaseReranker):
        """Assigns a fixed rerank_score of 99.0 to every candidate."""
        def rerank(self, query, candidates, top_k):
            return [
                c.model_copy(update={"rerank_score": 99.0, "final_score": 99.0})
                for c in candidates[:top_k]
            ]

    pipeline = QueryPipeline(
        keyword_index=bm25, vector_index=faiss,
        embedding_provider=provider, trace_store=trace_store,
        reranker=ScoringReranker(),
    )

    result = pipeline.query("retrieval")
    assert result.error is None
    assert len(result.candidates) > 0
    # Every candidate must carry the custom rerank score
    for cand in result.candidates:
        assert cand.rerank_score == pytest.approx(99.0)
        assert cand.final_score == pytest.approx(99.0)


def test_citations_reflect_reranked_order(ingested_env):
    """Citations must be built from the reranked (final) candidate order."""
    bm25, faiss, trace_store, provider = ingested_env
    pipeline = QueryPipeline(
        keyword_index=bm25,
        vector_index=faiss,
        embedding_provider=provider,
        trace_store=trace_store,
        reranker=NoOpReranker(),
    )
    result = pipeline.query("search results")
    for i, (cit, cand) in enumerate(zip(result.citations, result.candidates), start=1):
        assert cit.ref_number == i
        assert cit.chunk_id == cand.chunk_id
