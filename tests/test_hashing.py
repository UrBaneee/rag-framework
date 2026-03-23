"""Tests for hashing utilities — Task 11.1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rag.core.utils.hashing import (
    block_hash,
    canonicalize,
    chunk_signature,
    file_fingerprint,
    fingerprint_bytes,
    make_doc_id,
)


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------


def test_canonicalize_collapses_whitespace():
    assert canonicalize("  hello   world  ") == "hello world"


def test_canonicalize_collapses_newlines():
    assert canonicalize("line1\n\nline2") == "line1 line2"


def test_canonicalize_tabs():
    assert canonicalize("a\t\tb") == "a b"


def test_canonicalize_empty():
    assert canonicalize("") == ""


def test_canonicalize_preserves_case():
    assert canonicalize("Hello World") == "Hello World"


def test_canonicalize_idempotent():
    text = "  some   text  "
    assert canonicalize(canonicalize(text)) == canonicalize(text)


# ---------------------------------------------------------------------------
# block_hash
# ---------------------------------------------------------------------------


def test_block_hash_deterministic():
    assert block_hash("hello world") == block_hash("hello world")


def test_block_hash_whitespace_invariant():
    # Extra whitespace should not change the hash
    assert block_hash("hello  world") == block_hash("hello world")
    assert block_hash("  hello world  ") == block_hash("hello world")


def test_block_hash_different_content():
    assert block_hash("hello") != block_hash("world")


def test_block_hash_returns_hex():
    h = block_hash("test")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_block_hash_case_sensitive():
    # Case is preserved by default — different case → different hash
    assert block_hash("Hello") != block_hash("hello")


# ---------------------------------------------------------------------------
# chunk_signature
# ---------------------------------------------------------------------------


def test_chunk_signature_deterministic():
    hashes = ["aaa", "bbb", "ccc"]
    assert chunk_signature(hashes) == chunk_signature(hashes)


def test_chunk_signature_order_matters():
    assert chunk_signature(["a", "b"]) != chunk_signature(["b", "a"])


def test_chunk_signature_single_block():
    h = block_hash("only block")
    sig = chunk_signature([h])
    assert len(sig) == 64


def test_chunk_signature_empty():
    sig = chunk_signature([])
    assert len(sig) == 64  # sha256 of empty string — valid


def test_chunk_signature_changes_with_block():
    base = [block_hash("a"), block_hash("b")]
    changed = [block_hash("a"), block_hash("CHANGED")]
    assert chunk_signature(base) != chunk_signature(changed)


# ---------------------------------------------------------------------------
# file_fingerprint and fingerprint_bytes
# ---------------------------------------------------------------------------


def test_file_fingerprint_deterministic(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_bytes(b"hello world")
    assert file_fingerprint(f) == file_fingerprint(f)


def test_file_fingerprint_matches_sha256(tmp_path):
    content = b"test content"
    f = tmp_path / "doc.txt"
    f.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert file_fingerprint(f) == expected


def test_file_fingerprint_changes_with_content(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_bytes(b"version 1")
    fp1 = file_fingerprint(f)
    f.write_bytes(b"version 2")
    fp2 = file_fingerprint(f)
    assert fp1 != fp2


def test_fingerprint_bytes_matches_file(tmp_path):
    content = b"some raw bytes"
    f = tmp_path / "doc.bin"
    f.write_bytes(content)
    assert fingerprint_bytes(content) == file_fingerprint(f)


def test_fingerprint_bytes_empty():
    sig = fingerprint_bytes(b"")
    assert len(sig) == 64


# ---------------------------------------------------------------------------
# make_doc_id
# ---------------------------------------------------------------------------


def test_make_doc_id_deterministic():
    d1 = make_doc_id("/path/to/doc.pdf", "abc123")
    d2 = make_doc_id("/path/to/doc.pdf", "abc123")
    assert d1 == d2


def test_make_doc_id_changes_with_path():
    d1 = make_doc_id("/path/a.pdf", "abc123")
    d2 = make_doc_id("/path/b.pdf", "abc123")
    assert d1 != d2


def test_make_doc_id_changes_with_hash():
    d1 = make_doc_id("/path/doc.pdf", "hash_v1")
    d2 = make_doc_id("/path/doc.pdf", "hash_v2")
    assert d1 != d2


def test_make_doc_id_returns_hex():
    d = make_doc_id("/some/path", "somehash")
    assert len(d) == 64
    assert all(c in "0123456789abcdef" for c in d)


# ---------------------------------------------------------------------------
# Ingest pipeline — fingerprint-based skip
# ---------------------------------------------------------------------------


def test_ingest_skips_unchanged_document(tmp_path):
    """Second ingest of the same unchanged file returns skipped=True."""
    from unittest.mock import MagicMock

    from rag.pipelines.ingest_pipeline import IngestPipeline

    # Write a sample file
    doc_file = tmp_path / "sample.txt"
    doc_file.write_text("# Title\n\nSome content here.", encoding="utf-8")

    db_path = tmp_path / "rag.db"

    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore

    doc_store = SQLiteDocStore(db_path)
    trace_store = SQLiteTraceStore(db_path)
    pipeline = IngestPipeline(doc_store=doc_store, trace_store=trace_store)

    # First ingest — should succeed
    result1 = pipeline.ingest(doc_file)
    assert result1.error is None
    assert result1.skipped is False
    assert result1.chunk_count > 0

    # Second ingest — same file, should be skipped
    result2 = pipeline.ingest(doc_file)
    assert result2.skipped is True
    assert result2.doc_id == result1.doc_id
    assert result2.error is None


def test_ingest_reruns_changed_document(tmp_path):
    """After modifying a file, the second ingest processes it fully."""
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.ingest_pipeline import IngestPipeline

    doc_file = tmp_path / "sample.txt"
    doc_file.write_text("# Version 1\n\nOriginal content.", encoding="utf-8")
    db_path = tmp_path / "rag.db"

    doc_store = SQLiteDocStore(db_path)
    trace_store = SQLiteTraceStore(db_path)
    pipeline = IngestPipeline(doc_store=doc_store, trace_store=trace_store)

    result1 = pipeline.ingest(doc_file)
    assert result1.skipped is False

    # Modify the file
    doc_file.write_text("# Version 2\n\nUpdated content.", encoding="utf-8")

    result2 = pipeline.ingest(doc_file)
    assert result2.skipped is False
    assert result2.doc_id != result1.doc_id   # different content → different id
