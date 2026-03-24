"""Tests for NotionConnector — Task 15.4.

All tests are fully offline: _notion_request is monkeypatched so no
network access or Notion credentials are required.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rag.infra.connectors.notion_connector import (
    NotionConnector,
    _block_to_text,
    _page_title,
    _rich_text_to_str,
    _stable_source_id,
    blocks_to_text,
)


# ---------------------------------------------------------------------------
# Fixtures — canned Notion API payloads
# ---------------------------------------------------------------------------

_PATCH = "rag.infra.connectors.notion_connector._notion_request"


def _rt(text: str) -> list[dict]:
    """Build a minimal rich_text array."""
    return [{"plain_text": text, "type": "text"}]


def _page(
    page_id: str,
    title: str,
    last_edited: str = "2024-01-02T10:00:00.000Z",
    url: str = "",
    parent_type: str = "workspace",
    database_id: str = "",
) -> dict:
    parent: dict = {"type": parent_type}
    if parent_type == "database_id":
        parent["database_id"] = database_id
    return {
        "object": "page",
        "id": page_id,
        "last_edited_time": last_edited,
        "created_time": "2024-01-01T00:00:00.000Z",
        "url": url or f"https://notion.so/{page_id.replace('-', '')}",
        "parent": parent,
        "properties": {
            "Name": {
                "type": "title",
                "title": _rt(title),
            }
        },
    }


def _block(block_type: str, text: str, extra: dict | None = None) -> dict:
    data: dict = {"rich_text": _rt(text)}
    if extra:
        data.update(extra)
    return {"type": block_type, "id": "blk-1", block_type: data}


def _db_query_response(pages: list[dict]) -> dict:
    return {"object": "list", "results": pages, "has_more": False}


def _blocks_response(blocks: list[dict], has_more: bool = False) -> dict:
    return {"object": "list", "results": blocks, "has_more": has_more}


def _search_response(pages: list[dict]) -> dict:
    return {"object": "list", "results": pages, "has_more": False}


def _users_me_response() -> dict:
    return {"object": "user", "type": "bot", "name": "MyBot", "bot": {"owner": {"type": "workspace"}}}


# ---------------------------------------------------------------------------
# Unit tests — rich_text / block helpers
# ---------------------------------------------------------------------------


class TestRichTextToStr:
    def test_single_element(self):
        assert _rich_text_to_str(_rt("Hello")) == "Hello"

    def test_multiple_elements(self):
        rt = [{"plain_text": "foo"}, {"plain_text": " bar"}]
        assert _rich_text_to_str(rt) == "foo bar"

    def test_empty_list(self):
        assert _rich_text_to_str([]) == ""


class TestBlockToText:
    def test_paragraph(self):
        b = _block("paragraph", "Some text")
        assert _block_to_text(b) == "Some text"

    def test_heading_1(self):
        b = _block("heading_1", "Title")
        assert _block_to_text(b) == "# Title"

    def test_heading_2(self):
        b = _block("heading_2", "Sub")
        assert _block_to_text(b) == "## Sub"

    def test_heading_3(self):
        b = _block("heading_3", "Minor")
        assert _block_to_text(b) == "### Minor"

    def test_bulleted_list(self):
        b = _block("bulleted_list_item", "item")
        assert _block_to_text(b) == "- item"

    def test_numbered_list(self):
        b = _block("numbered_list_item", "step")
        assert _block_to_text(b) == "1. step"

    def test_quote(self):
        b = _block("quote", "wise words")
        assert _block_to_text(b) == "> wise words"

    def test_code(self):
        b = _block("code", "x = 1", extra={"language": "python"})
        result = _block_to_text(b)
        assert result.startswith("```python")
        assert "x = 1" in result

    def test_to_do_unchecked(self):
        b = _block("to_do", "buy milk", extra={"checked": False})
        assert _block_to_text(b) == "[ ] buy milk"

    def test_to_do_checked(self):
        b = _block("to_do", "done", extra={"checked": True})
        assert _block_to_text(b) == "[x] done"

    def test_child_page(self):
        b = {"type": "child_page", "child_page": {"title": "Subpage"}}
        assert _block_to_text(b) == "[Page: Subpage]"

    def test_unsupported_type_returns_none(self):
        b = {"type": "image", "image": {}}
        assert _block_to_text(b) is None

    def test_empty_rich_text_returns_none(self):
        b = {"type": "paragraph", "paragraph": {"rich_text": []}}
        assert _block_to_text(b) is None


class TestBlocksToText:
    def test_multiple_blocks(self):
        blocks = [
            _block("heading_1", "Title"),
            _block("paragraph", "Body paragraph"),
            _block("bulleted_list_item", "Point A"),
        ]
        text = blocks_to_text(blocks)
        lines = text.splitlines()
        assert lines[0] == "# Title"
        assert lines[1] == "Body paragraph"
        assert lines[2] == "- Point A"

    def test_skips_unsupported_blocks(self):
        blocks = [
            _block("paragraph", "Keep"),
            {"type": "image", "image": {}},
            _block("paragraph", "Also keep"),
        ]
        text = blocks_to_text(blocks)
        assert "Keep" in text
        assert "Also keep" in text
        assert text.count("\n") == 1  # exactly 2 lines

    def test_empty_list(self):
        assert blocks_to_text([]) == ""


class TestPageTitle:
    def test_extracts_title(self):
        page = _page("p1", "My Page Title")
        assert _page_title(page) == "My Page Title"

    def test_untitled_fallback(self):
        page = {"properties": {}}
        assert _page_title(page) == "(Untitled)"


class TestStableSourceId:
    def test_format(self):
        assert _stable_source_id("abc-123") == "notion:abc-123"

    def test_deterministic(self):
        assert _stable_source_id("x") == _stable_source_id("x")


# ---------------------------------------------------------------------------
# Integration tests — NotionConnector with mocked _notion_request
# ---------------------------------------------------------------------------


class TestNotionConnector:
    def _conn(self, db_ids: list[str] | None = None) -> NotionConnector:
        return NotionConnector(
            token="secret_fake",
            database_ids=db_ids or ["DB001"],
            fetch_blocks=False,  # most tests don't need block content
        )

    # ------ basic fetch -------------------------------------------------------

    def test_list_items_returns_artifacts(self):
        pages = [_page("P1", "Doc One", "2024-01-02T10:00:00.000Z")]
        with patch(_PATCH, return_value=_db_query_response(pages)):
            arts = self._conn().list_items()
        assert len(arts) == 1
        a = arts[0]
        assert a.source_type == "notion"
        assert a.source_id == "notion:P1"
        assert a.metadata["title"] == "Doc One"

    def test_empty_database_returns_empty(self):
        with patch(_PATCH, return_value=_db_query_response([])):
            arts = self._conn().list_items()
        assert arts == []

    def test_metadata_fields_populated(self):
        p = _page("P2", "Report", "2024-03-01T08:00:00.000Z", parent_type="database_id", database_id="DB001")
        with patch(_PATCH, return_value=_db_query_response([p])):
            arts = self._conn(["DB001"]).list_items()
        m = arts[0].metadata
        assert m["page_id"] == "P2"
        assert m["last_edited_time"] == "2024-03-01T08:00:00.000Z"
        assert m["parent_type"] == "database_id"
        assert m["parent_id"] == "DB001"

    # ------ cursor behaviour --------------------------------------------------

    def test_cursor_advances_to_newest_edited_time(self):
        pages = [
            _page("P1", "Old", "2024-01-01T00:00:00.000Z"),
            _page("P2", "New", "2024-06-15T12:00:00.000Z"),
        ]
        with patch(_PATCH, return_value=_db_query_response(pages)):
            conn = self._conn()
            conn.list_items()
        assert conn.next_cursor() == "2024-06-15T12:00:00.000Z"

    def test_cursor_empty_before_first_call(self):
        assert self._conn().next_cursor() == ""

    def test_since_cursor_passed_to_filter(self):
        with patch(_PATCH, return_value=_db_query_response([])) as mock_req:
            conn = self._conn()
            conn.list_items(since_cursor="2024-05-01T00:00:00.000Z")
        body = mock_req.call_args[1].get("body") or mock_req.call_args[0][3]
        assert body["filter"]["last_edited_time"]["after"] == "2024-05-01T00:00:00.000Z"

    def test_no_filter_when_cursor_empty(self):
        with patch(_PATCH, return_value=_db_query_response([])) as mock_req:
            conn = self._conn()
            conn.list_items(since_cursor="")
        body = mock_req.call_args[1].get("body") or mock_req.call_args[0][3]
        assert "filter" not in body

    # ------ block content -----------------------------------------------------

    def test_block_content_included_when_fetch_blocks_true(self):
        pages = [_page("P1", "Title")]
        blocks = [_block("paragraph", "Block body text")]

        call_count = [0]

        def fake_req(method, path, token, body=None, **kw):
            call_count[0] += 1
            if "databases" in path:
                return _db_query_response(pages)
            if "children" in path:
                return _blocks_response(blocks)
            return {}

        conn = NotionConnector(token="secret_fake", database_ids=["DB1"], fetch_blocks=True)
        with patch(_PATCH, side_effect=fake_req):
            arts = conn.list_items()

        assert "Block body text" in arts[0].content_text
        assert "Title" in arts[0].content_text

    def test_block_fetch_error_falls_back_to_title(self):
        pages = [_page("P1", "Fallback Title")]

        def fake_req(method, path, token, body=None, **kw):
            if "databases" in path:
                return _db_query_response(pages)
            raise RuntimeError("blocks fetch failed")

        conn = NotionConnector(token="secret_fake", database_ids=["DB1"], fetch_blocks=True)
        with patch(_PATCH, side_effect=fake_req):
            arts = conn.list_items()

        assert arts[0].content_text == "Fallback Title"

    # ------ multi-database ----------------------------------------------------

    def test_multiple_databases_synced(self):
        p1 = _page("P1", "From DB1", "2024-01-01T00:00:00.000Z")
        p2 = _page("P2", "From DB2", "2024-01-02T00:00:00.000Z")

        call_count = [0]

        def fake_req(method, path, token, body=None, **kw):
            call_count[0] += 1
            if "DB001" in path:
                return _db_query_response([p1])
            if "DB002" in path:
                return _db_query_response([p2])
            return _db_query_response([])

        with patch(_PATCH, side_effect=fake_req):
            conn = NotionConnector(token="secret_fake", database_ids=["DB001", "DB002"], fetch_blocks=False)
            arts = conn.list_items()

        assert len(arts) == 2
        page_ids = {a.metadata["page_id"] for a in arts}
        assert page_ids == {"P1", "P2"}

    def test_database_error_does_not_abort_others(self):
        p = _page("P2", "OK Page")

        def fake_req(method, path, token, body=None, **kw):
            if "DB001" in path:
                raise RuntimeError("db not found")
            return _db_query_response([p])

        with patch(_PATCH, side_effect=fake_req):
            conn = NotionConnector(token="secret_fake", database_ids=["DB001", "DB002"], fetch_blocks=False)
            arts = conn.list_items()

        assert len(arts) == 1
        assert arts[0].metadata["page_id"] == "P2"

    # ------ search fallback ---------------------------------------------------

    def test_search_api_used_when_no_database_ids(self):
        pages = [_page("P1", "Searched Page")]
        with patch(_PATCH, return_value=_search_response(pages)) as mock_req:
            conn = NotionConnector(token="secret_fake", database_ids=[], fetch_blocks=False)
            arts = conn.list_items()
        assert len(arts) == 1
        called_path = mock_req.call_args[0][1]
        assert called_path == "/search"

    # ------ healthcheck -------------------------------------------------------

    def test_healthcheck_ok(self):
        with patch(_PATCH, return_value=_users_me_response()):
            conn = self._conn()
            hc = conn.healthcheck()
        assert hc["status"] == "ok"
        assert hc["connector"] == "notion"

    def test_healthcheck_missing_token(self):
        conn = NotionConnector(token="", database_ids=[])
        hc = conn.healthcheck()
        assert hc["status"] == "error"
        assert "RAG_NOTION_TOKEN" in hc["detail"]

    def test_healthcheck_api_error(self):
        with patch(_PATCH, side_effect=RuntimeError("unauthorized")):
            conn = self._conn()
            hc = conn.healthcheck()
        assert hc["status"] == "error"
        assert "unauthorized" in hc["detail"]

    # ------ source_id stability -----------------------------------------------

    def test_source_id_stable_across_calls(self):
        pages = [_page("PAGE-XYZ", "Stable")]
        with patch(_PATCH, return_value=_db_query_response(pages)):
            arts1 = self._conn().list_items()
        with patch(_PATCH, return_value=_db_query_response(pages)):
            arts2 = self._conn().list_items()
        assert arts1[0].source_id == arts2[0].source_id

    def test_external_url_set(self):
        pages = [_page("P1", "Title", url="https://notion.so/P1")]
        with patch(_PATCH, return_value=_db_query_response(pages)):
            arts = self._conn().list_items()
        assert arts[0].external_url == "https://notion.so/P1"

    # ------ block pagination --------------------------------------------------

    def test_paginated_blocks_fetched(self):
        pages = [_page("P1", "Title")]
        page1_blocks = [_block("paragraph", "First")]
        page2_blocks = [_block("paragraph", "Second")]

        call_count = [0]

        def fake_req(method, path, token, body=None, **kw):
            call_count[0] += 1
            if "databases" in path:
                return _db_query_response(pages)
            if "children" in path:
                if call_count[0] == 2:
                    return {"results": page1_blocks, "has_more": True, "next_cursor": "cur2"}
                return {"results": page2_blocks, "has_more": False}
            return {}

        conn = NotionConnector(token="secret_fake", database_ids=["DB1"], fetch_blocks=True)
        with patch(_PATCH, side_effect=fake_req):
            arts = conn.list_items()

        text = arts[0].content_text
        assert "First" in text
        assert "Second" in text
