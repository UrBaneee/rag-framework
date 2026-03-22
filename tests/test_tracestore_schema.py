"""Unit tests for SQLite TraceStore schema creation."""

import sqlite3

import pytest

from rag.infra.stores.tracestore_sqlite import get_indexes, get_tables, init_schema


@pytest.mark.unit
def test_schema_creates_all_tables(tmp_path):
    db = tmp_path / "trace.sqlite"
    init_schema(db)
    tables = get_tables(db)
    assert "runs" in tables
    assert "trace_events" in tables


@pytest.mark.unit
def test_schema_creates_required_indexes(tmp_path):
    db = tmp_path / "trace.sqlite"
    init_schema(db)
    indexes = get_indexes(db)
    assert "idx_runs_run_type" in indexes
    assert "idx_runs_created_at" in indexes
    assert "idx_trace_events_run_id" in indexes
    assert "idx_trace_events_event_type" in indexes


@pytest.mark.unit
def test_schema_is_idempotent(tmp_path):
    db = tmp_path / "trace.sqlite"
    init_schema(db)
    init_schema(db)  # Must not raise
    assert set(get_tables(db)) == {"runs", "trace_events"}


@pytest.mark.unit
def test_schema_creates_parent_dirs(tmp_path):
    db = tmp_path / "subdir" / "trace.sqlite"
    init_schema(db)
    assert db.exists()


@pytest.mark.unit
def test_foreign_key_cascade_on_run_delete(tmp_path):
    db = tmp_path / "trace.sqlite"
    init_schema(db)

    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            "INSERT INTO runs(run_id, run_type) VALUES (?, ?)",
            ("run-1", "query"),
        )
        conn.execute(
            "INSERT INTO trace_events(event_id, run_id, event_type) VALUES (?, ?, ?)",
            ("evt-1", "run-1", "retrieval"),
        )
        conn.commit()

        conn.execute("DELETE FROM runs WHERE run_id = 'run-1'")
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM trace_events WHERE run_id = 'run-1'"
        ).fetchone()[0]
        assert count == 0
