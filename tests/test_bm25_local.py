"""Tests for BM25LocalIndex."""

import pytest

from rag.core.contracts.candidate import RetrievalSource
from rag.core.contracts.chunk import Chunk
from rag.infra.indexes.bm25_local import BM25LocalIndex


def _make_chunk(chunk_id: str, stable_text: str, doc_id: str = "doc1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        stable_text=stable_text,
        display_text=stable_text,
        chunk_signature=chunk_id,
    )


@pytest.fixture()
def five_chunks() -> list[Chunk]:
    return [
        _make_chunk("c1", "Python is a popular programming language"),
        _make_chunk("c2", "Machine learning uses statistical methods"),
        _make_chunk("c3", "Python supports machine learning libraries"),
        _make_chunk("c4", "Database indexing improves query performance"),
        _make_chunk("c5", "Natural language processing with Python"),
    ]


@pytest.fixture()
def index(five_chunks) -> BM25LocalIndex:
    idx = BM25LocalIndex()
    idx.add(five_chunks)
    return idx


# ── Basic search ───────────────────────────────────────────────────────────────

def test_search_returns_candidates(index):
    results = index.search("Python", top_k=3)
    assert len(results) == 3


def test_search_top_result_is_relevant(index):
    results = index.search("Python programming language", top_k=5)
    chunk_ids = [r.chunk_id for r in results]
    # c1 matches "Python programming language" best
    assert chunk_ids[0] == "c1"


def test_search_sets_bm25_score(index):
    results = index.search("machine learning", top_k=2)
    assert all(r.bm25_score is not None for r in results)
    assert all(r.bm25_score >= 0 for r in results)


def test_search_sets_retrieval_source(index):
    results = index.search("Python", top_k=1)
    assert results[0].retrieval_source == RetrievalSource.BM25


def test_search_respects_top_k(index):
    results = index.search("Python", top_k=2)
    assert len(results) <= 2


def test_search_empty_index_returns_empty():
    idx = BM25LocalIndex()
    assert idx.search("anything", top_k=5) == []


# ── Save and reload ───────────────────────────────────────────────────────────

def test_save_and_load_preserves_search(index, tmp_path):
    index.save(str(tmp_path))

    idx2 = BM25LocalIndex()
    idx2.load(str(tmp_path))

    results = idx2.search("Python programming language", top_k=5)
    assert results[0].chunk_id == "c1"


def test_save_creates_pkl_file(index, tmp_path):
    index.save(str(tmp_path))
    assert (tmp_path / "bm25.pkl").exists()


def test_load_missing_file_raises(tmp_path):
    idx = BM25LocalIndex()
    with pytest.raises(FileNotFoundError):
        idx.load(str(tmp_path))


# ── Remove ────────────────────────────────────────────────────────────────────

def test_remove_chunk_no_longer_in_results(index):
    index.remove("c1")
    results = index.search("Python programming language", top_k=5)
    chunk_ids = [r.chunk_id for r in results]
    assert "c1" not in chunk_ids


def test_remove_nonexistent_chunk_does_not_raise(index):
    # Should log a warning but not raise
    index.remove("no_such_id")
    assert len(index.search("Python", top_k=5)) == 5  # all still present
