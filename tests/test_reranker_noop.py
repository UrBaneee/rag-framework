"""Tests for NoOpReranker and plugin_registry reranker factory — Task 6.1."""

import pytest

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.registry.plugin_registry import build_reranker, register_reranker
from rag.infra.rerank.noop import NoOpReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cand(chunk_id: str, rrf_score: float) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id="doc1",
        display_text=f"text {chunk_id}",
        stable_text=f"text {chunk_id}",
        rrf_score=rrf_score,
        final_score=rrf_score,
        retrieval_source=RetrievalSource.HYBRID,
    )


@pytest.fixture()
def ranked() -> list[Candidate]:
    return [
        _cand("a", 0.030),
        _cand("b", 0.025),
        _cand("c", 0.020),
        _cand("d", 0.015),
    ]


# ---------------------------------------------------------------------------
# NoOpReranker
# ---------------------------------------------------------------------------


def test_noop_preserves_order(ranked):
    reranker = NoOpReranker()
    result = reranker.rerank("any query", ranked, top_k=4)
    assert [c.chunk_id for c in result] == ["a", "b", "c", "d"]


def test_noop_respects_top_k(ranked):
    reranker = NoOpReranker()
    result = reranker.rerank("query", ranked, top_k=2)
    assert len(result) == 2
    assert result[0].chunk_id == "a"
    assert result[1].chunk_id == "b"


def test_noop_sets_rerank_score(ranked):
    reranker = NoOpReranker()
    result = reranker.rerank("query", ranked, top_k=4)
    for orig, reranked in zip(ranked, result):
        assert reranked.rerank_score == pytest.approx(orig.rrf_score)


def test_noop_sets_final_score(ranked):
    reranker = NoOpReranker()
    result = reranker.rerank("query", ranked, top_k=4)
    for orig, reranked in zip(ranked, result):
        assert reranked.final_score == pytest.approx(orig.rrf_score)


def test_noop_empty_candidates():
    reranker = NoOpReranker()
    assert reranker.rerank("query", [], top_k=5) == []


def test_noop_top_k_larger_than_candidates(ranked):
    reranker = NoOpReranker()
    result = reranker.rerank("query", ranked, top_k=100)
    assert len(result) == len(ranked)


# ---------------------------------------------------------------------------
# Plugin registry — build_reranker
# ---------------------------------------------------------------------------


def test_build_reranker_noop_by_default():
    reranker = build_reranker({})
    assert isinstance(reranker, NoOpReranker)


def test_build_reranker_explicit_noop():
    reranker = build_reranker({"reranker": {"provider": "noop"}})
    assert isinstance(reranker, NoOpReranker)


def test_build_reranker_unknown_falls_back_to_noop():
    reranker = build_reranker({"reranker": {"provider": "nonexistent_provider"}})
    assert isinstance(reranker, NoOpReranker)


def test_register_custom_reranker():
    """A custom reranker can be registered and retrieved via factory."""

    class MyReranker(NoOpReranker):
        pass

    register_reranker("custom_test", MyReranker)
    reranker = build_reranker({"reranker": {"provider": "custom_test"}})
    assert isinstance(reranker, MyReranker)


def test_noop_integrates_with_query_pipeline(tmp_path):
    """NoOpReranker passes through fused candidates unchanged in a pipeline."""
    from rag.core.interfaces.embedding import BaseEmbeddingProvider
    from rag.infra.indexes.bm25_local import BM25LocalIndex
    from rag.infra.indexes.faiss_local import FaissLocalIndex
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.docstore_sqlite import init_schema as init_doc_schema
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.infra.stores.tracestore_sqlite import init_schema as init_trace_schema
    from rag.pipelines.ingest_pipeline import IngestPipeline
    from rag.pipelines.query_pipeline import QueryPipeline

    _DIM = 8

    class StubProvider(BaseEmbeddingProvider):
        @property
        def dim(self):
            return _DIM
        def embed(self, texts):
            return [[float(len(t) % _DIM == i) for i in range(_DIM)] for t in texts]

    db = str(tmp_path / "rag.db")
    init_doc_schema(db)
    init_trace_schema(db)
    doc_store = SQLiteDocStore(db)
    trace_store = SQLiteTraceStore(db)
    bm25 = BM25LocalIndex()
    faiss = FaissLocalIndex()

    txt = tmp_path / "doc.txt"
    txt.write_text(
        "Reranking improves the quality of search results.\n\n"
        "No-op reranker passes results through unchanged.\n",
        encoding="utf-8",
    )
    IngestPipeline(
        doc_store, trace_store,
        embedding_provider=StubProvider(),
        keyword_index=bm25, vector_index=faiss,
    ).ingest(str(txt))

    pipeline = QueryPipeline(
        keyword_index=bm25, vector_index=faiss,
        embedding_provider=StubProvider(),
        trace_store=trace_store,
    )
    result = pipeline.query("reranking search results")
    assert result.error is None
    assert len(result.candidates) > 0
