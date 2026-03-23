"""Tests for retrieval evaluation metrics — Task 10.1."""

import math

import pytest

from rag.pipelines.scoring.metrics import (
    AggregateMetrics,
    QueryMetrics,
    compute_aggregate_metrics,
    dcg_at_k,
    ideal_dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_perfect():
    assert recall_at_k(["c1", "c2", "c3"], ["c1", "c2", "c3"], k=3) == 1.0


def test_recall_partial():
    # 1 of 2 relevant items in top 3
    assert recall_at_k(["c1", "c4", "c5"], ["c1", "c2"], k=3) == pytest.approx(0.5)


def test_recall_none_found():
    assert recall_at_k(["c4", "c5"], ["c1", "c2"], k=2) == 0.0


def test_recall_empty_relevant():
    assert recall_at_k(["c1", "c2"], [], k=3) == 0.0


def test_recall_empty_retrieved():
    assert recall_at_k([], ["c1", "c2"], k=3) == 0.0


def test_recall_k_zero():
    assert recall_at_k(["c1", "c2"], ["c1"], k=0) == 0.0


def test_recall_k_smaller_than_list():
    # Only top-2 considered; c3 (relevant) is at rank 3 → not counted
    assert recall_at_k(["c4", "c5", "c3"], ["c3"], k=2) == 0.0


def test_recall_k_larger_than_list():
    # All retrieved considered
    assert recall_at_k(["c1"], ["c1", "c2"], k=10) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# reciprocal_rank (MRR per query)
# ---------------------------------------------------------------------------


def test_rr_first_rank():
    assert reciprocal_rank(["c1", "c2", "c3"], ["c1"], k=5) == pytest.approx(1.0)


def test_rr_second_rank():
    assert reciprocal_rank(["c4", "c1", "c3"], ["c1"], k=5) == pytest.approx(0.5)


def test_rr_third_rank():
    assert reciprocal_rank(["c4", "c5", "c1"], ["c1"], k=5) == pytest.approx(1 / 3)


def test_rr_not_found():
    assert reciprocal_rank(["c4", "c5"], ["c1"], k=2) == 0.0


def test_rr_beyond_k():
    # c1 is at rank 3, but k=2
    assert reciprocal_rank(["c4", "c5", "c1"], ["c1"], k=2) == 0.0


def test_rr_empty_relevant():
    assert reciprocal_rank(["c1"], [], k=5) == 0.0


def test_rr_multiple_relevant_uses_first():
    # Both c2 (rank 2) and c3 (rank 3) are relevant — should return 1/2
    assert reciprocal_rank(["c4", "c2", "c3"], ["c2", "c3"], k=5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# dcg_at_k and ideal_dcg_at_k
# ---------------------------------------------------------------------------


def test_dcg_single_hit_at_rank1():
    # rel=1 at rank 1 → 1/log2(2) = 1.0
    assert dcg_at_k(["c1"], ["c1"], k=1) == pytest.approx(1.0)


def test_dcg_single_hit_at_rank2():
    # rel=1 at rank 2 → 1/log2(3)
    assert dcg_at_k(["c4", "c1"], ["c1"], k=2) == pytest.approx(1 / math.log2(3))


def test_dcg_no_hits():
    assert dcg_at_k(["c4", "c5"], ["c1"], k=5) == 0.0


def test_ideal_dcg_two_relevant():
    expected = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ideal_dcg_at_k(["c1", "c2"], k=2) == pytest.approx(expected)


def test_ideal_dcg_k_smaller_than_relevant():
    # Only top-1 considered even if 2 relevant
    assert ideal_dcg_at_k(["c1", "c2"], k=1) == pytest.approx(1.0 / math.log2(2))


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_perfect():
    # Retrieved in ideal order
    assert ndcg_at_k(["c1", "c2"], ["c1", "c2"], k=2) == pytest.approx(1.0)


def test_ndcg_partial():
    # c1 retrieved at rank 1 (of 2 relevant)
    result = ndcg_at_k(["c1", "c4"], ["c1", "c2"], k=2)
    dcg = 1.0 / math.log2(2)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert result == pytest.approx(dcg / idcg)


def test_ndcg_no_hits():
    assert ndcg_at_k(["c4", "c5"], ["c1", "c2"], k=2) == 0.0


def test_ndcg_empty_relevant():
    assert ndcg_at_k(["c1"], [], k=3) == 0.0


def test_ndcg_between_zero_and_one():
    result = ndcg_at_k(["c3", "c1", "c4"], ["c1", "c2"], k=3)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# compute_aggregate_metrics
# ---------------------------------------------------------------------------


def test_aggregate_empty_results():
    agg = compute_aggregate_metrics([], k=10)
    assert isinstance(agg, AggregateMetrics)
    assert agg.num_queries == 0
    assert agg.mean_recall_at_k == 0.0
    assert agg.mrr == 0.0
    assert agg.mean_ndcg_at_k == 0.0


def test_aggregate_single_perfect_query():
    results = [{"retrieved": ["c1", "c2", "c3"], "relevant": ["c1", "c2", "c3"]}]
    agg = compute_aggregate_metrics(results, k=3)
    assert agg.num_queries == 1
    assert agg.mean_recall_at_k == pytest.approx(1.0)
    assert agg.mrr == pytest.approx(1.0)
    assert agg.mean_ndcg_at_k == pytest.approx(1.0)


def test_aggregate_single_miss_query():
    results = [{"retrieved": ["c4", "c5"], "relevant": ["c1", "c2"]}]
    agg = compute_aggregate_metrics(results, k=5)
    assert agg.mean_recall_at_k == 0.0
    assert agg.mrr == 0.0
    assert agg.mean_ndcg_at_k == 0.0


def test_aggregate_two_queries_averaging():
    results = [
        {"retrieved": ["c1", "c2"], "relevant": ["c1"], "query_id": "q1"},
        {"retrieved": ["c4", "c5"], "relevant": ["c1"], "query_id": "q2"},
    ]
    agg = compute_aggregate_metrics(results, k=5)
    assert agg.num_queries == 2
    # q1: recall=1.0, mrr=1.0 | q2: recall=0.0, mrr=0.0
    assert agg.mean_recall_at_k == pytest.approx(0.5)
    assert agg.mrr == pytest.approx(0.5)


def test_aggregate_per_query_populated():
    results = [
        {"retrieved": ["c1"], "relevant": ["c1"], "query_id": "q1"},
        {"retrieved": ["c4"], "relevant": ["c1"], "query_id": "q2"},
    ]
    agg = compute_aggregate_metrics(results, k=3)
    assert len(agg.per_query) == 2
    assert agg.per_query[0].query_id == "q1"
    assert agg.per_query[0].recall_at_k == pytest.approx(1.0)
    assert agg.per_query[1].query_id == "q2"
    assert agg.per_query[1].recall_at_k == 0.0


def test_aggregate_k_forwarded():
    results = [{"retrieved": ["c1"], "relevant": ["c1"]}]
    agg = compute_aggregate_metrics(results, k=7)
    assert agg.k == 7
    assert agg.per_query[0].k == 7


def test_aggregate_invalid_k_raises():
    with pytest.raises(ValueError):
        compute_aggregate_metrics([], k=0)


def test_aggregate_synthetic_eval_set():
    """Realistic tiny eval set: 5 queries with varying recall."""
    results = [
        {"retrieved": ["a", "b", "c", "d", "e"], "relevant": ["a", "c"], "query_id": "q1"},
        {"retrieved": ["x", "b", "c", "d", "e"], "relevant": ["b", "e"], "query_id": "q2"},
        {"retrieved": ["x", "y", "z"], "relevant": ["a", "b", "c"], "query_id": "q3"},
        {"retrieved": ["a", "b", "c"], "relevant": ["a", "b", "c"], "query_id": "q4"},
        {"retrieved": ["d", "e", "a"], "relevant": ["a"], "query_id": "q5"},
    ]
    agg = compute_aggregate_metrics(results, k=5)
    assert agg.num_queries == 5
    assert 0.0 <= agg.mean_recall_at_k <= 1.0
    assert 0.0 <= agg.mrr <= 1.0
    assert 0.0 <= agg.mean_ndcg_at_k <= 1.0
    # q4 is perfect → at least one query with recall=1.0
    assert any(q.recall_at_k == pytest.approx(1.0) for q in agg.per_query)
    # q3 has no hits → recall=0
    assert agg.per_query[2].recall_at_k == 0.0
