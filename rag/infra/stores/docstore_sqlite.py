"""SQLite implementation of the DocStore — schema creation."""

import sqlite3
from pathlib import Path


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    mime_type       TEXT NOT NULL DEFAULT '',
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    fingerprint     TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_TEXT_BLOCKS = """
CREATE TABLE IF NOT EXISTS text_blocks (
    block_id          TEXT PRIMARY KEY,
    doc_id            TEXT NOT NULL,
    block_type        TEXT NOT NULL DEFAULT 'paragraph',
    text              TEXT NOT NULL,
    block_hash        TEXT NOT NULL,
    page              INTEGER,
    sequence          INTEGER NOT NULL,
    section_path_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
"""

_DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id            TEXT PRIMARY KEY,
    doc_id              TEXT NOT NULL,
    stable_text         TEXT NOT NULL,
    display_text        TEXT NOT NULL,
    chunk_signature     TEXT NOT NULL,
    block_hashes_json   TEXT NOT NULL DEFAULT '[]',
    token_count         INTEGER NOT NULL DEFAULT 0,
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
"""

_DDL_INDEXES = [
    # text_blocks
    "CREATE INDEX IF NOT EXISTS idx_text_blocks_doc_id ON text_blocks(doc_id);",
    "CREATE INDEX IF NOT EXISTS idx_text_blocks_block_hash ON text_blocks(block_hash);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_text_blocks_doc_seq ON text_blocks(doc_id, sequence);",
    # chunks
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_signature ON chunks(chunk_signature);",
]


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_schema(db_path: str | Path) -> None:
    """Create the DocStore schema in a SQLite database.

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
            conn.execute(_DDL_DOCUMENTS)
            conn.execute(_DDL_TEXT_BLOCKS)
            conn.execute(_DDL_CHUNKS)
            for ddl in _DDL_INDEXES:
                conn.execute(ddl)
            conn.commit()
    except sqlite3.Error as exc:
        raise RuntimeError(f"Failed to initialise DocStore schema at {db_path}: {exc}") from exc


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
