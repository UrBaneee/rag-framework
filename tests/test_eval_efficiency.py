"""Tests for system efficiency metrics in eval pipeline — Task 10.3.

Covers:
- token_saved_est computed from candidate_tokens / packed_tokens
- mean_ingest_latency_ms forwarded from ingest_latency_ms parameter
- mean_query_latency_ms averaged across query entries
- skipped_chunks and changed_chunks are always None (pre Task 11.2)
- Fields are None when inputs are not supplied
- as_dict includes all efficiency keys
"""

from __future__ import annotations

import pytest

from rag.core.contracts.eval_report import EfficiencyMetrics
from rag.pipelines.eval_pipeline import run_eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_ENTRY = {"retrieved": ["c1"], "relevant": ["c1"]}


# ---------------------------------------------------------------------------
# token_saved_est
# ---------------------------------------------------------------------------


def test_token_saved_est_single_query():
    results = [
        {**_BASE_ENTRY, "candidate_tokens": 800, "packed_tokens": 300},
    ]
    report = run_eval(results, k=3)
    assert report.efficiency.token_saved_est == pytest.approx(500.0)


def test_token_saved_est_mean_across_queries():
    results = [
        {**_BASE_ENTRY, "candidate_tokens": 600, "packed_tokens": 200},  # saved 400
        {**_BASE_ENTRY, "candidate_tokens": 400, "packed_tokens": 200},  # saved 200
    ]
    report = run_eval(results, k=3)
    # mean(400, 200) == 300
    assert report.efficiency.token_saved_est == pytest.approx(300.0)


def test_token_saved_est_none_when_missing():
    results = [{**_BASE_ENTRY}]
    report = run_eval(results, k=3)
    assert report.efficiency.token_saved_est is None


def test_token_saved_est_partial_entries_ignored():
    """Only entries with both candidate_tokens and packed_tokens contribute."""
    results = [
        {**_BASE_ENTRY, "candidate_tokens": 1000, "packed_tokens": 400},  # saved 600
        {**_BASE_ENTRY},  # no token data — skipped
    ]
    report = run_eval(results, k=3)
    assert report.efficiency.token_saved_est == pytest.approx(600.0)


def test_token_saved_est_zero_when_packed_equals_candidates():
    results = [
        {**_BASE_ENTRY, "candidate_tokens": 500, "packed_tokens": 500},
    ]
    report = run_eval(results, k=3)
    assert report.efficiency.token_saved_est == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ingest_latency_ms
# ---------------------------------------------------------------------------


def test_ingest_latency_forwarded():
    results = [{**_BASE_ENTRY}]
    report = run_eval(results, k=3, ingest_latency_ms=120.5)
    assert report.efficiency.mean_ingest_latency_ms == pytest.approx(120.5)


def test_ingest_latency_none_by_default():
    results = [{**_BASE_ENTRY}]
    report = run_eval(results, k=3)
    assert report.efficiency.mean_ingest_latency_ms is None


def test_ingest_latency_empty_results():
    report = run_eval([], k=3, ingest_latency_ms=55.0)
    # Empty results short-circuits; efficiency defaults are None
    assert report.efficiency.mean_ingest_latency_ms is None


# ---------------------------------------------------------------------------
# mean_query_latency_ms
# ---------------------------------------------------------------------------


def test_query_latency_single():
    results = [{**_BASE_ENTRY, "query_latency_ms": 42.0}]
    report = run_eval(results, k=3)
    assert report.efficiency.mean_query_latency_ms == pytest.approx(42.0)


def test_query_latency_mean_across_queries():
    results = [
        {**_BASE_ENTRY, "query_latency_ms": 30.0},
        {**_BASE_ENTRY, "query_latency_ms": 70.0},
    ]
    report = run_eval(results, k=3)
    assert report.efficiency.mean_query_latency_ms == pytest.approx(50.0)


def test_query_latency_none_when_missing():
    results = [{**_BASE_ENTRY}]
    report = run_eval(results, k=3)
    assert report.efficiency.mean_query_latency_ms is None


def test_query_latency_partial_entries():
    """Only entries that provide query_latency_ms contribute to the mean."""
    results = [
        {**_BASE_ENTRY, "query_latency_ms": 60.0},
        {**_BASE_ENTRY},  # no latency — skipped
    ]
    report = run_eval(results, k=3)
    assert report.efficiency.mean_query_latency_ms == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# skipped_chunks / changed_chunks always None (pre Task 11.2)
# ---------------------------------------------------------------------------


def test_skipped_chunks_is_none():
    results = [{**_BASE_ENTRY}]
    report = run_eval(results, k=3)
    assert report.efficiency.skipped_chunks is None


def test_changed_chunks_is_none():
    results = [{**_BASE_ENTRY}]
    report = run_eval(results, k=3)
    assert report.efficiency.changed_chunks is None


# ---------------------------------------------------------------------------
# as_dict includes efficiency keys
# ---------------------------------------------------------------------------


def test_as_dict_includes_efficiency():
    results = [
        {**_BASE_ENTRY, "query_latency_ms": 25.0, "candidate_tokens": 500, "packed_tokens": 200},
    ]
    report = run_eval(results, k=3, ingest_latency_ms=80.0)
    d = report.as_dict()
    assert "efficiency" in d
    eff = d["efficiency"]
    assert "token_saved_est" in eff
    assert "mean_ingest_latency_ms" in eff
    assert "mean_query_latency_ms" in eff
    assert "skipped_chunks" in eff
    assert "changed_chunks" in eff
    assert eff["token_saved_est"] == pytest.approx(300.0)
    assert eff["mean_ingest_latency_ms"] == pytest.approx(80.0)
    assert eff["mean_query_latency_ms"] == pytest.approx(25.0)
    assert eff["skipped_chunks"] is None
    assert eff["changed_chunks"] is None


def test_efficiency_metrics_as_dict_standalone():
    em = EfficiencyMetrics(
        token_saved_est=150.0,
        mean_ingest_latency_ms=200.0,
        mean_query_latency_ms=35.0,
    )
    d = em.as_dict()
    assert d["token_saved_est"] == pytest.approx(150.0)
    assert d["mean_ingest_latency_ms"] == pytest.approx(200.0)
    assert d["mean_query_latency_ms"] == pytest.approx(35.0)
    assert d["skipped_chunks"] is None
    assert d["changed_chunks"] is None


# ---------------------------------------------------------------------------
# Realistic scenario: two ingest runs with one small change
# ---------------------------------------------------------------------------


def test_realistic_two_ingest_runs():
    """Simulate ingest twice, query twice — verify all efficiency fields present."""
    # First ingest: 300 ms, second ingest: 280 ms — we pass the second latency
    results = [
        {
            "query_id": "q1",
            "retrieved": ["c1", "c2", "c3"],
            "relevant": ["c1", "c3"],
            "query_latency_ms": 35.0,
            "candidate_tokens": 900,
            "packed_tokens": 300,
        },
        {
            "query_id": "q2",
            "retrieved": ["c4", "c5"],
            "relevant": ["c4"],
            "query_latency_ms": 28.0,
            "candidate_tokens": 600,
            "packed_tokens": 200,
        },
    ]
    report = run_eval(results, k=5, ingest_latency_ms=280.0)

    assert report.num_queries == 2
    eff = report.efficiency

    # Ingest latency
    assert eff.mean_ingest_latency_ms == pytest.approx(280.0)

    # Query latency: mean(35, 28) == 31.5
    assert eff.mean_query_latency_ms == pytest.approx(31.5)

    # Token savings: mean(900-300, 600-200) == mean(600, 400) == 500
    assert eff.token_saved_est == pytest.approx(500.0)

    # Block-diff fields still None
    assert eff.skipped_chunks is None
    assert eff.changed_chunks is None
