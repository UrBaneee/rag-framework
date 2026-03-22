"""SQLite implementation of the TraceStore — schema creation."""

import sqlite3
from pathlib import Path


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    run_type    TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_TRACE_EVENTS = """
CREATE TABLE IF NOT EXISTS trace_events (
    event_id    TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
"""

_DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_runs_run_type ON runs(run_type);",
    "CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_trace_events_run_id ON trace_events(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_trace_events_event_type ON trace_events(event_type);",
]


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_schema(db_path: str | Path) -> None:
    """Create the TraceStore schema in a SQLite database.

    Safe to call on an existing database — all statements use
    ``CREATE TABLE IF NOT EXISTS`` and ``CREATE INDEX IF NOT EXISTS``.

    Args:
        db_path: Path to the SQLite database file. The file and any
            parent directories are created if they do not exist.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute(_DDL_RUNS)
            conn.execute(_DDL_TRACE_EVENTS)
            for ddl in _DDL_INDEXES:
                conn.execute(ddl)
            conn.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Failed to initialise TraceStore schema at {db_path}: {exc}"
        ) from exc


def get_tables(db_path: str | Path) -> list[str]:
    """Return names of all user tables in the database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Sorted list of table names.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
    return [row[0] for row in rows]


def get_indexes(db_path: str | Path) -> list[str]:
    """Return names of all indexes in the database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Sorted list of index names.
    """
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name;"
        ).fetchall()
    return [row[0] for row in rows]
