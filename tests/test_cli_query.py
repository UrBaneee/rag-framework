"""Tests for rag.cli.query — Task 5.4."""

from pathlib import Path

import pytest

from rag.cli.query import main
from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.infra.indexes.bm25_local import BM25LocalIndex
from rag.infra.indexes.faiss_local import FaissLocalIndex
from rag.infra.stores.docstore_sqlite import SQLiteDocStore, init_schema as init_doc_schema
from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore, init_schema as init_trace_schema
from rag.pipelines.ingest_pipeline import IngestPipeline

_DIM = 8


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
def ingested_env(tmp_path: Path):
    """Ingest a sample file and save indexes; return (db_path, index_dir)."""
    db_path = tmp_path / "data" / "default.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "data" / "default_indexes"

    doc_store, trace_store = _make_stores(str(db_path))
    bm25 = BM25LocalIndex()
    faiss = FaissLocalIndex()

    txt = tmp_path / "doc.txt"
    txt.write_text(
        "Retrieval augmented generation combines search with language models.\n\n"
        "BM25 is a keyword-based retrieval algorithm used in search engines.\n\n"
        "Vector search finds semantically similar documents using dense embeddings.\n",
        encoding="utf-8",
    )

    pipeline = IngestPipeline(
        doc_store,
        trace_store,
        embedding_provider=StubEmbeddingProvider(),
        keyword_index=bm25,
        vector_index=faiss,
    )
    result = pipeline.ingest(str(txt))
    assert result.error is None

    # Persist indexes so CLI can load them
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25.save(str(index_dir))
    faiss.save(str(index_dir))

    return db_path, index_dir


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def test_cli_query_exits_zero(ingested_env):
    db_path, index_dir = ingested_env
    code = main([
        "retrieval augmented generation",
        "--db", str(db_path),
        "--index-dir", str(index_dir),
    ])
    assert code == 0


def test_cli_query_prints_results(ingested_env, capsys):
    db_path, index_dir = ingested_env
    main([
        "BM25 search",
        "--db", str(db_path),
        "--index-dir", str(index_dir),
    ])
    out = capsys.readouterr().out
    assert "Results for:" in out
    assert "result(s)" in out


def test_cli_query_prints_citations(ingested_env, capsys):
    db_path, index_dir = ingested_env
    main([
        "vector embeddings",
        "--db", str(db_path),
        "--index-dir", str(index_dir),
    ])
    out = capsys.readouterr().out
    # At least one citation marker [1] should appear
    assert "[1]" in out


def test_cli_query_top_k_respected(ingested_env, capsys):
    db_path, index_dir = ingested_env
    main([
        "search language model",
        "--db", str(db_path),
        "--index-dir", str(index_dir),
        "--top-k", "1",
    ])
    out = capsys.readouterr().out
    # [2] should not appear when top_k=1
    assert "[1]" in out
    assert "[2]" not in out


def test_cli_query_verbose_shows_scores(ingested_env, capsys):
    db_path, index_dir = ingested_env
    main([
        "retrieval",
        "--db", str(db_path),
        "--index-dir", str(index_dir),
        "--verbose",
    ])
    out = capsys.readouterr().out
    assert "rrf=" in out
    assert "source=" in out


def test_cli_query_missing_db_returns_error(tmp_path):
    code = main([
        "any query",
        "--db", str(tmp_path / "nonexistent.db"),
        "--index-dir", str(tmp_path / "indexes"),
    ])
    assert code == 1
