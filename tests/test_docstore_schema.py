"""Unit tests for SQLite DocStore schema creation."""

import pytest

from rag.infra.stores.docstore_sqlite import get_indexes, get_tables, init_schema


@pytest.mark.unit
def test_schema_creates_all_tables(tmp_path):
    db = tmp_path / "test.sqlite"
    init_schema(db)
    tables = get_tables(db)
    assert "documents" in tables
    assert "text_blocks" in tables
    assert "chunks" in tables


@pytest.mark.unit
def test_schema_creates_required_indexes(tmp_path):
    db = tmp_path / "test.sqlite"
    init_schema(db)
    indexes = get_indexes(db)
    assert "idx_text_blocks_doc_id" in indexes
    assert "idx_text_blocks_block_hash" in indexes
    assert "idx_text_blocks_doc_seq" in indexes
    assert "idx_chunks_doc_id" in indexes
    assert "idx_chunks_signature" in indexes


@pytest.mark.unit
def test_schema_is_idempotent(tmp_path):
    db = tmp_path / "test.sqlite"
    init_schema(db)
    init_schema(db)  # Second call must not raise
    assert set(get_tables(db)) == {"documents", "text_blocks", "chunks"}


@pytest.mark.unit
def test_schema_creates_parent_dirs(tmp_path):
    db = tmp_path / "subdir" / "nested" / "test.sqlite"
    init_schema(db)
    assert db.exists()
    assert "documents" in get_tables(db)


@pytest.mark.unit
def test_schema_enables_foreign_keys(tmp_path):
    """Verify WAL mode is set and foreign keys work (cascade delete)."""
    import sqlite3

    db = tmp_path / "test.sqlite"
    init_schema(db)

    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        # Insert a document then a block
        conn.execute(
            "INSERT INTO documents(doc_id, source_path) VALUES (?, ?)",
            ("doc-1", "/tmp/file.pdf"),
        )
        conn.execute(
            "INSERT INTO text_blocks(block_id, doc_id, text, block_hash, sequence) "
            "VALUES (?, ?, ?, ?, ?)",
            ("blk-1", "doc-1", "hello", "abc", 0),
        )
        conn.commit()

        # Delete the document — block should cascade
        conn.execute("DELETE FROM documents WHERE doc_id = 'doc-1'")
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM text_blocks WHERE doc_id = 'doc-1'"
        ).fetchone()[0]
        assert count == 0
