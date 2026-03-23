"""Tests for RRFFusion — Task 5.2."""

import pytest

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.infra.indexes.rrf_fusion import RRFFusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cand(
    chunk_id: str,
    bm25_score: float | None = None,
    vector_score: float | None = None,
    source: RetrievalSource = RetrievalSource.BM25,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id="doc1",
        display_text=f"text {chunk_id}",
        stable_text=f"text {chunk_id}",
        bm25_score=bm25_score,
        vector_score=vector_score,
        retrieval_source=source,
    )


def _bm25_list(*ids: str) -> list[Candidate]:
    return [_cand(cid, bm25_score=float(10 - i)) for i, cid in enumerate(ids)]


def _vec_list(*ids: str) -> list[Candidate]:
    return [
        _cand(cid, vector_score=float(-(i * 0.1)), source=RetrievalSource.VECTOR)
        for i, cid in enumerate(ids)
    ]


# ---------------------------------------------------------------------------
# Basic fusion
# ---------------------------------------------------------------------------


def test_fuse_non_overlapping_all_present():
    """All unique chunk_ids from both lists must appear in the result."""
    bm25 = _bm25_list("a", "b")
    vec = _vec_list("c", "d")
    fusion = RRFFusion()
    result = fusion.fuse([bm25, vec])
    ids = {c.chunk_id for c in result}
    assert ids == {"a", "b", "c", "d"}


def test_fuse_overlapping_chunk_ranks_higher():
    """A chunk in both lists should outscore chunks in only one list."""
    bm25 = _bm25_list("shared", "bm25_only")
    vec = _vec_list("shared", "vec_only")
    fusion = RRFFusion()
    result = fusion.fuse([bm25, vec])
    scores = {c.chunk_id: c.rrf_score for c in result}
    assert scores["shared"] > scores["bm25_only"]
    assert scores["shared"] > scores["vec_only"]


def test_fuse_result_ordered_descending():
    """Result list must be sorted by descending rrf_score."""
    bm25 = _bm25_list("a", "b", "c")
    vec = _vec_list("b", "c", "d")
    fusion = RRFFusion()
    result = fusion.fuse([bm25, vec])
    scores = [c.rrf_score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_fuse_is_deterministic():
    """Calling fuse twice with identical input must return identical order."""
    bm25 = _bm25_list("x", "y", "z")
    vec = _vec_list("z", "x", "w")
    fusion = RRFFusion()
    r1 = [c.chunk_id for c in fusion.fuse([bm25, vec])]
    r2 = [c.chunk_id for c in fusion.fuse([bm25, vec])]
    assert r1 == r2


def test_fuse_rrf_score_populated():
    bm25 = _bm25_list("a")
    fusion = RRFFusion()
    result = fusion.fuse([bm25])
    assert result[0].rrf_score > 0
    assert result[0].final_score == result[0].rrf_score


# ---------------------------------------------------------------------------
# RRF score correctness
# ---------------------------------------------------------------------------


def test_rrf_score_formula():
    """Score for rank-1 in a single list with k=60 should be 1/(60+1)."""
    bm25 = [_cand("a", bm25_score=5.0)]
    fusion = RRFFusion(k=60)
    result = fusion.fuse([bm25])
    assert result[0].rrf_score == pytest.approx(1 / 61)


def test_rrf_score_sum_across_lists():
    """Chunk at rank 1 in both lists: score = 1/(k+1) + 1/(k+1)."""
    k = 60
    bm25 = [_cand("a", bm25_score=5.0)]
    vec = [_cand("a", vector_score=-0.1, source=RetrievalSource.VECTOR)]
    fusion = RRFFusion(k=k)
    result = fusion.fuse([bm25, vec])
    expected = 2 * (1 / (k + 1))
    assert result[0].rrf_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Source attribution after fusion
# ---------------------------------------------------------------------------


def test_source_hybrid_when_in_all_lists():
    bm25 = _bm25_list("shared")
    vec = _vec_list("shared")
    fusion = RRFFusion()
    result = fusion.fuse([bm25, vec])
    assert result[0].retrieval_source == RetrievalSource.HYBRID


def test_source_preserved_when_single_list():
    bm25 = _bm25_list("only_bm25")
    vec = _vec_list("only_vec")
    fusion = RRFFusion()
    result = {c.chunk_id: c for c in fusion.fuse([bm25, vec])}
    assert result["only_bm25"].retrieval_source == RetrievalSource.BM25
    assert result["only_vec"].retrieval_source == RetrievalSource.VECTOR


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_lists_return_empty():
    fusion = RRFFusion()
    assert fusion.fuse([[], []]) == []


def test_single_list_works():
    bm25 = _bm25_list("a", "b", "c")
    fusion = RRFFusion()
    result = fusion.fuse([bm25])
    assert len(result) == 3
    assert result[0].chunk_id == "a"  # rank-1 has highest score


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        RRFFusion(k=0)


def test_empty_ranked_lists_raises():
    fusion = RRFFusion()
    with pytest.raises(ValueError):
        fusion.fuse([])
