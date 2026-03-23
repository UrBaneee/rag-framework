"""Tests for the eval CLI entry point — Task 10.4."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.cli.eval import (
    _case_outcome,
    _load_suite,
    _resolve_suite_path,
    main,
)


# ---------------------------------------------------------------------------
# _resolve_suite_path
# ---------------------------------------------------------------------------


def test_resolve_suite_path_default(tmp_path):
    path = _resolve_suite_path("example_queries", suite_file=None)
    assert path.name == "example_queries.json"


def test_resolve_suite_path_override(tmp_path):
    custom = tmp_path / "custom.json"
    result = _resolve_suite_path("example_queries", suite_file=custom)
    assert result == custom.resolve()


# ---------------------------------------------------------------------------
# _load_suite
# ---------------------------------------------------------------------------


def test_load_suite_json(tmp_path):
    data = [{"query": "What is RAG?", "expected_behavior": "answer"}]
    f = tmp_path / "suite.json"
    f.write_text(json.dumps(data))
    loaded = _load_suite(f)
    assert len(loaded) == 1
    assert loaded[0]["query"] == "What is RAG?"


def test_load_suite_jsonl(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text(
        '{"query": "q1"}\n{"query": "q2"}\n'
    )
    loaded = _load_suite(f)
    assert len(loaded) == 2
    assert loaded[1]["query"] == "q2"


def test_load_suite_empty_lines_ignored(tmp_path):
    f = tmp_path / "suite.jsonl"
    f.write_text('{"query": "q1"}\n\n{"query": "q2"}\n')
    loaded = _load_suite(f)
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# _case_outcome
# ---------------------------------------------------------------------------


def test_case_outcome_pass():
    entry = {"retrieved": ["c1", "c2"], "relevant": ["c1"]}
    assert _case_outcome(entry, k=5) == "PASS"


def test_case_outcome_fail():
    entry = {"retrieved": ["c3", "c4"], "relevant": ["c1"]}
    assert _case_outcome(entry, k=5) == "FAIL"


def test_case_outcome_skip_no_relevant():
    entry = {"retrieved": ["c1"], "relevant": []}
    assert _case_outcome(entry, k=5) == "SKIP"


def test_case_outcome_k_cutoff():
    # c1 is relevant but at rank 3, k=2 → FAIL
    entry = {"retrieved": ["c2", "c3", "c1"], "relevant": ["c1"]}
    assert _case_outcome(entry, k=2) == "FAIL"


def test_case_outcome_k_cutoff_pass():
    entry = {"retrieved": ["c1", "c2", "c3"], "relevant": ["c1"]}
    assert _case_outcome(entry, k=2) == "PASS"


# ---------------------------------------------------------------------------
# main — --answer-quality stub
# ---------------------------------------------------------------------------


def test_main_answer_quality_returns_zero(capsys):
    rc = main(["--answer-quality"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "RAGAS not available" in captured.out


# ---------------------------------------------------------------------------
# main — missing suite file
# ---------------------------------------------------------------------------


def test_main_missing_suite_file_returns_one(tmp_path, capsys):
    # Point to a non-existent file
    rc = main(["--suite", "example_queries",
               "--suite-file", str(tmp_path / "nonexistent.json")])
    assert rc == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


# ---------------------------------------------------------------------------
# main — suite found but DB missing
# ---------------------------------------------------------------------------


def test_main_missing_db_returns_one(tmp_path, capsys):
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps([{"query": "What is RAG?"}]))

    rc = main([
        "--suite", "example_queries",
        "--suite-file", str(suite_file),
        "--db", str(tmp_path / "nonexistent.db"),
    ])
    assert rc == 1
    captured = capsys.readouterr()
    assert "database not found" in captured.err or "not found" in captured.err


# ---------------------------------------------------------------------------
# main — empty suite
# ---------------------------------------------------------------------------


def test_main_empty_suite_returns_zero(tmp_path, capsys):
    suite_file = tmp_path / "suite.json"
    suite_file.write_text("[]")
    rc = main(["--suite", "example_queries", "--suite-file", str(suite_file)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "empty" in captured.out


# ---------------------------------------------------------------------------
# main — full run with mocked pipeline
# ---------------------------------------------------------------------------


def _make_fake_candidate(chunk_id: str, source_label: str = "bm25_only"):
    cand = MagicMock()
    cand.chunk_id = chunk_id
    cand.source_label = source_label
    cand.bm25_score = 1.0
    cand.vector_score = None
    return cand


def _make_fake_query_result(chunk_ids: list[str]):
    qr = MagicMock()
    qr.candidates = [_make_fake_candidate(cid) for cid in chunk_ids]
    qr.error = None
    return qr


def test_main_full_run_with_mocked_pipeline(tmp_path, capsys):
    """Full run: suite file + DB both exist, pipeline mocked."""
    db_path = tmp_path / "test.db"
    db_path.touch()
    index_dir = tmp_path / "idx"
    index_dir.mkdir()

    suite = [
        {"query": "What is RAG?", "expected_sources": ["c1"], "expected_behavior": "answer"},
        {"query": "How does BM25 work?", "expected_sources": [], "expected_behavior": "abstain"},
    ]
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps(suite))

    # Patch the entire _run_queries to avoid real pipeline init
    mock_results = [
        {
            "query_id": "What is RAG?",
            "query": "What is RAG?",
            "retrieved": ["c1", "c2"],
            "relevant": ["c1"],
            "candidates": [_make_fake_candidate("c1"), _make_fake_candidate("c2")],
            "expected_behavior": "answer",
            "query_latency_ms": 10.0,
            "error": None,
        },
        {
            "query_id": "How does BM25 work?",
            "query": "How does BM25 work?",
            "retrieved": ["c3"],
            "relevant": [],
            "candidates": [_make_fake_candidate("c3")],
            "expected_behavior": "abstain",
            "query_latency_ms": 8.0,
            "error": None,
        },
    ]

    with patch("rag.cli.eval._run_queries", return_value=mock_results):
        rc = main([
            "--suite", "example_queries",
            "--suite-file", str(suite_file),
            "--db", str(db_path),
            "--index-dir", str(index_dir),
            "--k", "5",
        ])

    assert rc == 0
    captured = capsys.readouterr()
    # Per-case output
    assert "What is RAG?" in captured.out
    assert "PASS" in captured.out
    assert "SKIP" in captured.out  # second entry has no expected_sources
    # Aggregate output
    assert "Aggregate metrics" in captured.out
    assert "Mean Recall" in captured.out
    assert "MRR" in captured.out


def test_main_all_fail_still_returns_zero(tmp_path, capsys):
    """All FAIL cases: exit 0 (failures are expected retrieval misses, not errors)."""
    db_path = tmp_path / "test.db"
    db_path.touch()
    suite_file = tmp_path / "suite.json"
    suite_file.write_text(json.dumps([{"query": "q1", "expected_sources": ["c1"]}]))

    mock_results = [
        {
            "query_id": "q1",
            "query": "q1",
            "retrieved": ["c9"],
            "relevant": ["c1"],
            "candidates": [],
            "expected_behavior": "answer",
            "query_latency_ms": 5.0,
            "error": None,
        }
    ]

    with patch("rag.cli.eval._run_queries", return_value=mock_results):
        rc = main([
            "--suite", "example_queries",
            "--suite-file", str(suite_file),
            "--db", str(db_path),
            "--k", "5",
        ])

    # Exit 0 — retrieval misses are expected failures, not crash errors
    assert rc == 0
    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    assert "0/1 passed" in captured.out or "failed" in captured.out
