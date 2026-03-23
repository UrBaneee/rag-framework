"""Tests for eval pipeline — Task 10.2.

Covers:
- run_eval returns EvalReport with correct retrieval metrics
- source attribution ratios (bm25_only, vector_only, both) are computed
- edge cases: empty dataset, missing candidates field, k validation
"""

from __future__ import annotations

import math

import pytest

from rag.core.contracts.eval_report import EvalReport, SourceAttributionStats
from rag.pipelines.eval_pipeline import run_eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(source_label: str):
    """Return a minimal object that exposes source_label."""

    class _Cand:
        pass

    c = _Cand()
    c.source_label = source_label
    return c


# ---------------------------------------------------------------------------
# Basic retrieval metrics
# ---------------------------------------------------------------------------


def test_run_eval_empty_returns_report():
    report = run_eval([], k=5)
    assert isinstance(report, EvalReport)
    assert report.num_queries == 0
    assert report.mean_recall_at_k == 0.0
    assert report.mrr == 0.0
    assert report.mean_ndcg_at_k == 0.0


def test_run_eval_single_perfect_query():
    results = [{"retrieved": ["c1", "c2"], "relevant": ["c1", "c2"], "query_id": "q1"}]
    report = run_eval(results, k=2)
    assert report.num_queries == 1
    assert report.mean_recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    assert report.mean_ndcg_at_k == pytest.approx(1.0)
    assert report.k == 2


def test_run_eval_single_miss_query():
    results = [{"retrieved": ["c3", "c4"], "relevant": ["c1", "c2"]}]
    report = run_eval(results, k=5)
    assert report.mean_recall_at_k == 0.0
    assert report.mrr == 0.0
    assert report.mean_ndcg_at_k == 0.0


def test_run_eval_two_queries_averaging():
    results = [
        {"retrieved": ["c1"], "relevant": ["c1"], "query_id": "q1"},
        {"retrieved": ["c3"], "relevant": ["c1"], "query_id": "q2"},
    ]
    report = run_eval(results, k=5)
    assert report.num_queries == 2
    assert report.mean_recall_at_k == pytest.approx(0.5)
    assert report.mrr == pytest.approx(0.5)


def test_run_eval_invalid_k_raises():
    with pytest.raises(ValueError):
        run_eval([], k=0)


def test_run_eval_k_forwarded():
    results = [{"retrieved": ["c1"], "relevant": ["c1"]}]
    report = run_eval(results, k=7)
    assert report.k == 7
    assert report.per_query[0].k == 7


# ---------------------------------------------------------------------------
# Source attribution ratios
# ---------------------------------------------------------------------------


def test_source_attribution_all_bm25():
    candidates = [_make_candidate("bm25_only")] * 4
    results = [
        {"retrieved": ["c1"], "relevant": ["c1"], "candidates": candidates}
    ]
    report = run_eval(results, k=5)
    sa = report.source_attribution
    assert sa.bm25_only == pytest.approx(1.0)
    assert sa.vector_only == pytest.approx(0.0)
    assert sa.both == pytest.approx(0.0)
    assert sa.total_candidates == 4


def test_source_attribution_all_vector():
    candidates = [_make_candidate("vector_only")] * 3
    results = [{"retrieved": [], "relevant": [], "candidates": candidates}]
    report = run_eval(results, k=5)
    sa = report.source_attribution
    assert sa.vector_only == pytest.approx(1.0)
    assert sa.bm25_only == pytest.approx(0.0)
    assert sa.both == pytest.approx(0.0)


def test_source_attribution_all_both():
    candidates = [_make_candidate("both")] * 5
    results = [{"retrieved": [], "relevant": [], "candidates": candidates}]
    report = run_eval(results, k=5)
    sa = report.source_attribution
    assert sa.both == pytest.approx(1.0)
    assert sa.bm25_only == pytest.approx(0.0)


def test_source_attribution_mixed():
    # 2 bm25_only, 1 vector_only, 1 both → total 4
    candidates = [
        _make_candidate("bm25_only"),
        _make_candidate("bm25_only"),
        _make_candidate("vector_only"),
        _make_candidate("both"),
    ]
    results = [{"retrieved": [], "relevant": [], "candidates": candidates}]
    report = run_eval(results, k=5)
    sa = report.source_attribution
    assert sa.total_candidates == 4
    assert sa.bm25_only == pytest.approx(0.5)
    assert sa.vector_only == pytest.approx(0.25)
    assert sa.both == pytest.approx(0.25)


def test_source_attribution_no_candidates():
    """When candidates list is absent, attribution defaults to zeros."""
    results = [{"retrieved": ["c1"], "relevant": ["c1"]}]
    report = run_eval(results, k=3)
    sa = report.source_attribution
    assert sa.total_candidates == 0
    assert sa.bm25_only == 0.0
    assert sa.vector_only == 0.0
    assert sa.both == 0.0


def test_source_attribution_across_multiple_queries():
    # q1: 2 bm25_only | q2: 2 vector_only → total 4, 50% / 50%
    results = [
        {
            "retrieved": [],
            "relevant": [],
            "candidates": [_make_candidate("bm25_only"), _make_candidate("bm25_only")],
        },
        {
            "retrieved": [],
            "relevant": [],
            "candidates": [_make_candidate("vector_only"), _make_candidate("vector_only")],
        },
    ]
    report = run_eval(results, k=5)
    sa = report.source_attribution
    assert sa.total_candidates == 4
    assert sa.bm25_only == pytest.approx(0.5)
    assert sa.vector_only == pytest.approx(0.5)
    assert sa.both == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Per-query breakdown
# ---------------------------------------------------------------------------


def test_per_query_source_counts():
    candidates = [
        _make_candidate("bm25_only"),
        _make_candidate("vector_only"),
        _make_candidate("both"),
    ]
    results = [
        {
            "retrieved": ["c1"],
            "relevant": ["c1"],
            "query_id": "q1",
            "candidates": candidates,
        }
    ]
    report = run_eval(results, k=3)
    q = report.per_query[0]
    assert q.query_id == "q1"
    assert q.bm25_only_count == 1
    assert q.vector_only_count == 1
    assert q.both_count == 1


def test_per_query_populated():
    results = [
        {"retrieved": ["c1"], "relevant": ["c1"], "query_id": "qa"},
        {"retrieved": ["c9"], "relevant": ["c1"], "query_id": "qb"},
    ]
    report = run_eval(results, k=3)
    assert len(report.per_query) == 2
    assert report.per_query[0].query_id == "qa"
    assert report.per_query[0].recall_at_k == pytest.approx(1.0)
    assert report.per_query[1].query_id == "qb"
    assert report.per_query[1].recall_at_k == 0.0


# ---------------------------------------------------------------------------
# as_dict
# ---------------------------------------------------------------------------


def test_eval_report_as_dict():
    results = [{"retrieved": ["c1"], "relevant": ["c1"]}]
    report = run_eval(results, k=3)
    d = report.as_dict()
    assert "mean_recall_at_k" in d
    assert "mrr" in d
    assert "mean_ndcg_at_k" in d
    assert "source_attribution" in d
    sa_d = d["source_attribution"]
    assert "bm25_only" in sa_d
    assert "vector_only" in sa_d
    assert "both" in sa_d


def test_source_attribution_stats_as_dict():
    sa = SourceAttributionStats(bm25_only=0.4, vector_only=0.3, both=0.3, total_candidates=10)
    d = sa.as_dict()
    assert d["bm25_only"] == pytest.approx(0.4)
    assert d["vector_only"] == pytest.approx(0.3)
    assert d["both"] == pytest.approx(0.3)
    assert d["total_candidates"] == 10


# ---------------------------------------------------------------------------
# Dict-based candidates (alternative interface)
# ---------------------------------------------------------------------------


def test_source_attribution_dict_candidates():
    """Candidates can be plain dicts with 'source_label' key."""
    candidates = [
        {"source_label": "bm25_only"},
        {"source_label": "both"},
    ]
    results = [{"retrieved": [], "relevant": [], "candidates": candidates}]
    report = run_eval(results, k=5)
    sa = report.source_attribution
    assert sa.total_candidates == 2
    assert sa.bm25_only == pytest.approx(0.5)
    assert sa.both == pytest.approx(0.5)
