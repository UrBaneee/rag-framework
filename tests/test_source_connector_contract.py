"""Tests for source connector interface and sync contract — Task 15.1.

Verifies:
- BaseSourceConnector ABC cannot be instantiated directly.
- A concrete FakeConnector satisfies the interface.
- SourceArtifact fields, helpers, and serialisation.
- DocStore cursor persistence (save / load / get_connector_state).
"""

from __future__ import annotations

import pytest

from rag.core.contracts.source_artifact import SourceArtifact
from rag.core.interfaces.source_connector import BaseSourceConnector


# ---------------------------------------------------------------------------
# Fake connector (minimal concrete implementation)
# ---------------------------------------------------------------------------

class FakeConnector(BaseSourceConnector):
    """Minimal connector for interface-contract tests."""

    connector_name = "fake"

    def __init__(self, items: list[SourceArtifact] | None = None) -> None:
        self._items = items or []
        self._cursor = ""

    def list_items(self, since_cursor: str = "") -> list[SourceArtifact]:
        self._cursor = f"cursor-after-{len(self._items)}"
        return list(self._items)

    def next_cursor(self) -> str:
        return self._cursor

    def healthcheck(self) -> dict:
        return {"status": "ok", "connector": self.connector_name, "detail": ""}


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------

class TestBaseSourceConnectorABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseSourceConnector()  # type: ignore[abstract]

    def test_fake_connector_is_instance(self):
        conn = FakeConnector()
        assert isinstance(conn, BaseSourceConnector)

    def test_list_items_returns_list(self):
        artifact = SourceArtifact(source_type="fake", source_id="a1", content_text="hello")
        conn = FakeConnector(items=[artifact])
        result = conn.list_items()
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].source_id == "a1"

    def test_next_cursor_after_list_items(self):
        conn = FakeConnector(items=[SourceArtifact(source_type="fake", source_id="x")])
        assert conn.next_cursor() == ""  # before any call
        conn.list_items()
        assert conn.next_cursor() != ""

    def test_healthcheck_structure(self):
        conn = FakeConnector()
        hc = conn.healthcheck()
        assert "status" in hc
        assert "connector" in hc
        assert "detail" in hc
        assert hc["status"] == "ok"
        assert hc["connector"] == "fake"

    def test_since_cursor_forwarded(self):
        """list_items should accept a since_cursor without raising."""
        conn = FakeConnector()
        result = conn.list_items(since_cursor="some-cursor-value")
        assert result == []


# ---------------------------------------------------------------------------
# SourceArtifact contract
# ---------------------------------------------------------------------------

class TestSourceArtifact:
    def test_required_fields(self):
        a = SourceArtifact(source_type="email", source_id="msg-001")
        assert a.source_type == "email"
        assert a.source_id == "msg-001"

    def test_default_fields(self):
        a = SourceArtifact(source_type="slack", source_id="ts-123")
        assert a.external_url == ""
        assert a.content_bytes is None
        assert a.content_text is None
        assert a.mime_type == "text/plain"
        assert a.metadata == {}
        assert a.cursor_after == ""

    def test_has_content_text(self):
        a = SourceArtifact(source_type="notion", source_id="p1", content_text="Hello")
        assert a.has_content() is True

    def test_has_content_bytes(self):
        a = SourceArtifact(source_type="gdocs", source_id="d1", content_bytes=b"pdf...")
        assert a.has_content() is True

    def test_has_content_empty(self):
        a = SourceArtifact(source_type="email", source_id="empty")
        assert a.has_content() is False

    def test_as_text_from_text(self):
        a = SourceArtifact(source_type="slack", source_id="s1", content_text="Hello world")
        assert a.as_text() == "Hello world"

    def test_as_text_from_bytes(self):
        a = SourceArtifact(source_type="email", source_id="e1", content_bytes=b"byte content")
        assert a.as_text() == "byte content"

    def test_as_text_text_takes_precedence_over_bytes(self):
        a = SourceArtifact(
            source_type="email",
            source_id="e2",
            content_text="text wins",
            content_bytes=b"bytes lose",
        )
        assert a.as_text() == "text wins"

    def test_as_text_empty(self):
        a = SourceArtifact(source_type="notion", source_id="n1")
        assert a.as_text() == ""

    def test_as_dict_keys(self):
        a = SourceArtifact(
            source_type="gdocs",
            source_id="doc-1",
            external_url="https://docs.google.com/...",
            content_text="some text",
            metadata={"title": "My Doc"},
            cursor_after="page-token-xyz",
        )
        d = a.as_dict()
        assert d["source_type"] == "gdocs"
        assert d["source_id"] == "doc-1"
        assert d["external_url"] == "https://docs.google.com/..."
        assert d["content_text_len"] == len("some text")
        assert d["content_bytes_len"] is None
        assert d["metadata"] == {"title": "My Doc"}
        assert d["cursor_after"] == "page-token-xyz"

    def test_metadata_stores_reference(self):
        """SourceArtifact stores the metadata dict by reference (no copy)."""
        meta = {"key": "val"}
        a = SourceArtifact(source_type="slack", source_id="s2", metadata=meta)
        assert a.metadata is meta  # same object — callers own the dict


# ---------------------------------------------------------------------------
# DocStore cursor persistence
# ---------------------------------------------------------------------------

class TestDocStoreCursorPersistence:
    @pytest.fixture()
    def store(self, tmp_path):
        from rag.infra.stores.docstore_sqlite import SQLiteDocStore, init_schema
        db = tmp_path / "test.db"
        init_schema(db)
        return SQLiteDocStore(db)

    def test_load_missing_cursor_returns_empty(self, store):
        assert store.load_connector_cursor("email") == ""

    def test_save_and_load_cursor(self, store):
        store.save_connector_cursor("slack", "ts-1234567890.000100")
        assert store.load_connector_cursor("slack") == "ts-1234567890.000100"

    def test_upsert_updates_cursor(self, store):
        store.save_connector_cursor("notion", "cursor-v1")
        store.save_connector_cursor("notion", "cursor-v2")
        assert store.load_connector_cursor("notion") == "cursor-v2"

    def test_different_connectors_independent(self, store):
        store.save_connector_cursor("email", "email-cursor")
        store.save_connector_cursor("slack", "slack-cursor")
        assert store.load_connector_cursor("email") == "email-cursor"
        assert store.load_connector_cursor("slack") == "slack-cursor"

    def test_get_connector_state_none_when_missing(self, store):
        assert store.get_connector_state("gdocs") is None

    def test_get_connector_state_returns_dict(self, store):
        store.save_connector_cursor("gdocs", "page-token-abc")
        state = store.get_connector_state("gdocs")
        assert state is not None
        assert state["connector_name"] == "gdocs"
        assert state["cursor"] == "page-token-abc"
        assert "last_sync_at" in state

    def test_connector_state_table_exists(self, tmp_path):
        from rag.infra.stores.docstore_sqlite import get_tables, init_schema
        db = tmp_path / "schema_check.db"
        init_schema(db)
        tables = get_tables(db)
        assert "connector_state" in tables
