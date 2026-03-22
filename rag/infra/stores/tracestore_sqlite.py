"""SQLite implementation of the TraceStore — schema creation and CRUD."""

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Optional

from rag.core.contracts.trace import AnswerTrace, PipelineStep
from rag.core.interfaces.trace_store import BaseTraceStore


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


# ── SQLiteTraceStore ───────────────────────────────────────────────────────────

class SQLiteTraceStore(BaseTraceStore):
    """SQLite-backed trace store.

    Initialises the schema on construction and exposes CRUD methods for
    pipeline runs and answer traces.

    AnswerTrace objects are serialised to a single trace_events row with
    ``event_type = "answer_trace"`` and the full trace JSON in ``data_json``.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        init_schema(self._db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    # ── Runs ───────────────────────────────────────────────────────────────────

    def save_run(self, run_type: str, metadata: dict[str, Any]) -> str:
        """Record a new pipeline run and return its run_id.

        Args:
            run_type: Pipeline type, e.g. "ingest" or "query".
            metadata: Arbitrary metadata about this run.

        Returns:
            A unique run_id string.
        """
        run_id = str(uuid.uuid4())
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO runs(run_id, run_type, metadata_json) VALUES (?, ?, ?)",
                    (run_id, run_type, json.dumps(metadata)),
                )
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to save run: {exc}") from exc
        return run_id

    def list_runs(
        self, run_type: Optional[str] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List recent pipeline runs, most recent first.

        Args:
            run_type: Filter by type or None for all.
            limit: Maximum number of runs to return.

        Returns:
            List of run metadata dicts.
        """
        with self._conn() as conn:
            if run_type is not None:
                rows = conn.execute(
                    "SELECT run_id, run_type, metadata_json, created_at "
                    "FROM runs WHERE run_type = ? ORDER BY created_at DESC LIMIT ?",
                    (run_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT run_id, run_type, metadata_json, created_at "
                    "FROM runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "run_id": row["run_id"],
                "run_type": row["run_type"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ── Answer traces ──────────────────────────────────────────────────────────

    def save_answer_trace(self, run_id: str, trace: AnswerTrace) -> None:
        """Persist a complete AnswerTrace for a query run.

        Args:
            run_id: Run identifier returned by ``save_run``.
            trace: The AnswerTrace to store.
        """
        event_id = str(uuid.uuid4())
        data = {
            "query": trace.query,
            "prompt_tokens": trace.prompt_tokens,
            "completion_tokens": trace.completion_tokens,
            "total_tokens": trace.total_tokens,
            "total_latency_ms": trace.total_latency_ms,
            "model": trace.model,
            "rerank_provider": trace.rerank_provider,
            "candidates_before_rerank": trace.candidates_before_rerank,
            "candidates_after_rerank": trace.candidates_after_rerank,
            "context_chunks_used": trace.context_chunks_used,
            "steps": [s.model_dump() for s in trace.steps],
        }
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO trace_events(event_id, run_id, event_type, data_json) "
                    "VALUES (?, ?, ?, ?)",
                    (event_id, run_id, "answer_trace", json.dumps(data)),
                )
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to save answer trace for run {run_id}: {exc}") from exc

    def get_answer_trace(self, run_id: str) -> Optional[AnswerTrace]:
        """Retrieve the AnswerTrace for a query run.

        Args:
            run_id: Run identifier.

        Returns:
            The AnswerTrace if found, or None.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data_json FROM trace_events "
                "WHERE run_id = ? AND event_type = 'answer_trace' "
                "ORDER BY created_at DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["data_json"])
        return AnswerTrace(
            query=data["query"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            total_tokens=data["total_tokens"],
            total_latency_ms=data["total_latency_ms"],
            model=data["model"],
            rerank_provider=data.get("rerank_provider"),
            candidates_before_rerank=data["candidates_before_rerank"],
            candidates_after_rerank=data["candidates_after_rerank"],
            context_chunks_used=data["context_chunks_used"],
            steps=[PipelineStep(**s) for s in data.get("steps", [])],
            run_id=run_id,
        )
