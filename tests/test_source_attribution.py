"""Tests for Task 5.1 — retrieval source attribution."""

import pytest

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.pipelines.query_pipeline import attribute_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bm25(chunk_id: str, score: float = 1.0) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id="doc1",
        display_text=f"text {chunk_id}",
        stable_text=f"text {chunk_id}",
        bm25_score=score,
        retrieval_source=RetrievalSource.BM25,
    )


def _vec(chunk_id: str, score: float = -0.1) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id="doc1",
        display_text=f"text {chunk_id}",
        stable_text=f"text {chunk_id}",
        vector_score=score,
        retrieval_source=RetrievalSource.VECTOR,
    )


# ---------------------------------------------------------------------------
# source_label property
# ---------------------------------------------------------------------------


def test_source_label_bm25_only():
    c = _bm25("c1")
    assert c.source_label == "bm25_only"


def test_source_label_vector_only():
    c = _vec("c1")
    assert c.source_label == "vector_only"


def test_source_label_both():
    c = Candidate(
        chunk_id="c1",
        doc_id="doc1",
        display_text="t",
        stable_text="t",
        bm25_score=1.0,
        vector_score=-0.1,
        retrieval_source=RetrievalSource.HYBRID,
    )
    assert c.source_label == "both"


# ---------------------------------------------------------------------------
# attribute_candidates merging
# ---------------------------------------------------------------------------


def test_non_overlapping_results():
    """No shared chunk_ids → all are single-source."""
    bm25 = [_bm25("a"), _bm25("b")]
    vec = [_vec("c"), _vec("d")]
    merged = attribute_candidates(bm25, vec)
    assert len(merged) == 4
    labels = {c.chunk_id: c.source_label for c in merged}
    assert labels["a"] == "bm25_only"
    assert labels["b"] == "bm25_only"
    assert labels["c"] == "vector_only"
    assert labels["d"] == "vector_only"


def test_fully_overlapping_results():
    """All chunk_ids shared → all marked 'both'."""
    bm25 = [_bm25("a"), _bm25("b")]
    vec = [_vec("a"), _vec("b")]
    merged = attribute_candidates(bm25, vec)
    assert len(merged) == 2
    for c in merged:
        assert c.source_label == "both"


def test_partial_overlap():
    """Mixed overlap: one shared, two single-source."""
    bm25 = [_bm25("shared"), _bm25("bm25_only")]
    vec = [_vec("shared"), _vec("vec_only")]
    merged = attribute_candidates(bm25, vec)
    assert len(merged) == 3
    labels = {c.chunk_id: c.source_label for c in merged}
    assert labels["shared"] == "both"
    assert labels["bm25_only"] == "bm25_only"
    assert labels["vec_only"] == "vector_only"


def test_overlap_merges_both_scores():
    """Merged 'both' candidate carries bm25_score and vector_score."""
    bm25 = [_bm25("x", score=2.5)]
    vec = [_vec("x", score=-0.05)]
    merged = attribute_candidates(bm25, vec)
    assert len(merged) == 1
    c = merged[0]
    assert c.bm25_score == pytest.approx(2.5)
    assert c.vector_score == pytest.approx(-0.05)


def test_empty_bm25():
    vec = [_vec("a"), _vec("b")]
    merged = attribute_candidates([], vec)
    assert len(merged) == 2
    assert all(c.source_label == "vector_only" for c in merged)


def test_empty_vector():
    bm25 = [_bm25("a"), _bm25("b")]
    merged = attribute_candidates(bm25, [])
    assert len(merged) == 2
    assert all(c.source_label == "bm25_only" for c in merged)


def test_both_empty():
    assert attribute_candidates([], []) == []


def test_retrieval_source_enum_set_correctly():
    """retrieval_source enum value must match source_label semantics."""
    bm25 = [_bm25("a")]
    vec = [_vec("b"), _vec("a")]
    merged = {c.chunk_id: c for c in attribute_candidates(bm25, vec)}

    assert merged["a"].retrieval_source == RetrievalSource.HYBRID
    assert merged["b"].retrieval_source == RetrievalSource.VECTOR
