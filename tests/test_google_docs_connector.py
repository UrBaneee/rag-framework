"""Tests for GoogleDocsConnector — Task 15.5.

All tests are fully offline: Drive and Docs API calls are monkeypatched
via the connector's internal _drive_get / _docs_get methods, or by
injecting a pre-minted _access_token so JWT minting is bypassed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.infra.connectors.google_docs_connector import (
    GoogleDocsConnector,
    _paragraph_to_text,
    _stable_source_id,
    _table_to_text,
    doc_content_to_text,
)


# ---------------------------------------------------------------------------
# Fixtures — canned API payloads
# ---------------------------------------------------------------------------


def _file(
    file_id: str,
    name: str,
    modified: str = "2024-03-01T10:00:00.000Z",
    created: str = "2024-01-01T00:00:00.000Z",
    web_link: str = "",
    owners: list[str] | None = None,
) -> dict:
    return {
        "id": file_id,
        "name": name,
        "modifiedTime": modified,
        "createdTime": created,
        "webViewLink": web_link or f"https://docs.google.com/document/d/{file_id}/edit",
        "owners": [{"emailAddress": e} for e in (owners or ["owner@example.com"])],
    }


def _drive_response(files: list[dict]) -> dict:
    return {"files": files}


def _para(text: str, style: str = "NORMAL_TEXT") -> dict:
    return {
        "paragraph": {
            "paragraphStyle": {"namedStyleType": style},
            "elements": [{"textRun": {"content": text}}],
        }
    }


def _heading(text: str, level: int = 1) -> dict:
    return _para(text, f"HEADING_{level}")


def _docs_response(file_id: str, title: str, content: list[dict]) -> dict:
    return {
        "documentId": file_id,
        "title": title,
        "body": {"content": content},
    }


def _make_conn(
    folder_ids: list[str] | None = None,
    fetch_content: bool = False,
) -> GoogleDocsConnector:
    return GoogleDocsConnector(
        folder_ids=folder_ids or [],
        fetch_content=fetch_content,
        _access_token="fake-token",  # bypass JWT minting
    )


# ---------------------------------------------------------------------------
# Unit tests — document content helpers
# ---------------------------------------------------------------------------


class TestParagraphToText:
    def test_normal_text(self):
        p = _para("Hello world")["paragraph"]
        assert _paragraph_to_text(p) == "Hello world"

    def test_heading_1(self):
        p = _heading("Chapter 1", 1)["paragraph"]
        assert _paragraph_to_text(p) == "# Chapter 1"

    def test_heading_2(self):
        p = _heading("Section", 2)["paragraph"]
        assert _paragraph_to_text(p) == "## Section"

    def test_heading_3(self):
        p = _heading("Sub", 3)["paragraph"]
        assert _paragraph_to_text(p) == "### Sub"

    def test_title_style(self):
        p = _para("Doc Title", "TITLE")["paragraph"]
        assert _paragraph_to_text(p) == "# Doc Title"

    def test_empty_paragraph_returns_none(self):
        p = {"paragraphStyle": {"namedStyleType": "NORMAL_TEXT"}, "elements": []}
        assert _paragraph_to_text(p) is None

    def test_whitespace_only_returns_none(self):
        p = _para("   \n")["paragraph"]
        assert _paragraph_to_text(p) is None

    def test_trailing_newline_stripped(self):
        p = _para("text\n")["paragraph"]
        assert _paragraph_to_text(p) == "text"


class TestTableToText:
    def test_simple_table(self):
        table = {
            "tableRows": [
                {
                    "tableCells": [
                        {"content": [_para("A")]},
                        {"content": [_para("B")]},
                    ]
                },
                {
                    "tableCells": [
                        {"content": [_para("C")]},
                        {"content": [_para("D")]},
                    ]
                },
            ]
        }
        text = _table_to_text(table)
        lines = text.splitlines()
        assert len(lines) == 2
        assert "A" in lines[0] and "B" in lines[0]
        assert "C" in lines[1] and "D" in lines[1]

    def test_empty_table(self):
        assert _table_to_text({"tableRows": []}) == ""


class TestDocContentToText:
    def test_mixed_content(self):
        content = [
            _heading("Title", 1),
            _para("First paragraph"),
            _para("Second paragraph"),
        ]
        text = doc_content_to_text(content)
        assert "# Title" in text
        assert "First paragraph" in text
        assert "Second paragraph" in text

    def test_skips_unknown_elements(self):
        content = [
            _para("Keep"),
            {"inlineObjectElement": {}},
            _para("Also keep"),
        ]
        text = doc_content_to_text(content)
        assert "Keep" in text
        assert "Also keep" in text

    def test_empty_content(self):
        assert doc_content_to_text([]) == ""


class TestStableSourceId:
    def test_format(self):
        assert _stable_source_id("doc-abc") == "gdocs:doc-abc"

    def test_deterministic(self):
        assert _stable_source_id("x") == _stable_source_id("x")


# ---------------------------------------------------------------------------
# Integration tests — GoogleDocsConnector
# ---------------------------------------------------------------------------


class TestGoogleDocsConnector:

    # ------ basic fetch (metadata only) -------------------------------------

    def test_list_items_returns_artifacts(self):
        files = [_file("F1", "My Doc")]
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response(files)):
            arts = conn.list_items()
        assert len(arts) == 1
        assert arts[0].source_type == "google_docs"
        assert arts[0].source_id == "gdocs:F1"

    def test_artifact_metadata_fields(self):
        f = _file("F2", "Report", "2024-04-01T08:00:00.000Z", owners=["user@x.com"])
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response([f])):
            arts = conn.list_items()
        m = arts[0].metadata
        assert m["file_id"] == "F2"
        assert m["name"] == "Report"
        assert m["modified_time"] == "2024-04-01T08:00:00.000Z"
        assert "user@x.com" in m["owners"]

    def test_empty_drive_returns_empty(self):
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response([])):
            arts = conn.list_items()
        assert arts == []

    # ------ cursor behaviour ------------------------------------------------

    def test_cursor_advances_to_newest_modified_time(self):
        files = [
            _file("F1", "Old", "2024-01-01T00:00:00.000Z"),
            _file("F2", "New", "2024-09-15T12:00:00.000Z"),
        ]
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response(files)):
            conn.list_items()
        assert conn.next_cursor() == "2024-09-15T12:00:00.000Z"

    def test_cursor_empty_before_first_call(self):
        assert _make_conn().next_cursor() == ""

    def test_since_cursor_included_in_query(self):
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response([])) as mock_dg:
            conn.list_items(since_cursor="2024-06-01T00:00:00.000Z")
        params = mock_dg.call_args[0][2]
        assert "2024-06-01T00:00:00.000Z" in params["q"]

    def test_no_time_filter_when_cursor_empty(self):
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response([])) as mock_dg:
            conn.list_items(since_cursor="")
        params = mock_dg.call_args[0][2]
        assert "modifiedTime" not in params["q"]

    # ------ folder filtering ------------------------------------------------

    def test_folder_id_included_in_query(self):
        conn = _make_conn(folder_ids=["FOLDER1"])
        with patch.object(conn, "_drive_get", return_value=_drive_response([])) as mock_dg:
            conn.list_items()
        params = mock_dg.call_args[0][2]
        assert "FOLDER1" in params["q"]

    def test_multiple_folder_ids_in_query(self):
        conn = _make_conn(folder_ids=["F1", "F2"])
        with patch.object(conn, "_drive_get", return_value=_drive_response([])) as mock_dg:
            conn.list_items()
        params = mock_dg.call_args[0][2]
        assert "F1" in params["q"] and "F2" in params["q"]

    # ------ document content ------------------------------------------------

    def test_doc_content_fetched_when_enabled(self):
        files = [_file("DOC1", "My Title")]
        doc = _docs_response("DOC1", "My Title", [_para("Body text")])

        conn = GoogleDocsConnector(
            folder_ids=[], fetch_content=True, _access_token="tok"
        )
        with patch.object(conn, "_drive_get", return_value=_drive_response(files)):
            with patch.object(conn, "_docs_get", return_value=doc):
                arts = conn.list_items()

        assert "Body text" in arts[0].content_text
        assert "My Title" in arts[0].content_text

    def test_heading_rendered_in_content(self):
        files = [_file("DOC2", "Doc")]
        doc = _docs_response("DOC2", "Doc", [_heading("Chapter", 1), _para("Para")])

        conn = GoogleDocsConnector(folder_ids=[], fetch_content=True, _access_token="tok")
        with patch.object(conn, "_drive_get", return_value=_drive_response(files)):
            with patch.object(conn, "_docs_get", return_value=doc):
                arts = conn.list_items()

        assert "# Chapter" in arts[0].content_text

    def test_content_fetch_error_falls_back_to_filename(self):
        files = [_file("DOC3", "Fallback Name")]

        conn = GoogleDocsConnector(folder_ids=[], fetch_content=True, _access_token="tok")
        with patch.object(conn, "_drive_get", return_value=_drive_response(files)):
            with patch.object(conn, "_docs_get", side_effect=RuntimeError("403")):
                arts = conn.list_items()

        assert arts[0].content_text == "Fallback Name"

    def test_no_docs_call_when_fetch_content_false(self):
        files = [_file("DOC4", "Skipped")]

        conn = _make_conn(fetch_content=False)
        with patch.object(conn, "_drive_get", return_value=_drive_response(files)):
            with patch.object(conn, "_docs_get") as mock_docs:
                conn.list_items()

        mock_docs.assert_not_called()

    # ------ external_url / source_id ----------------------------------------

    def test_external_url_set_from_web_view_link(self):
        f = _file("D1", "Doc", web_link="https://docs.google.com/document/d/D1/edit")
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response([f])):
            arts = conn.list_items()
        assert arts[0].external_url == "https://docs.google.com/document/d/D1/edit"

    def test_source_id_stable_across_calls(self):
        files = [_file("STABLE1", "S")]
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response(files)):
            arts1 = conn.list_items()
        with patch.object(conn, "_drive_get", return_value=_drive_response(files)):
            arts2 = conn.list_items()
        assert arts1[0].source_id == arts2[0].source_id

    # ------ healthcheck -----------------------------------------------------

    def test_healthcheck_ok(self):
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value={"files": []}):
            hc = conn.healthcheck()
        assert hc["status"] == "ok"
        assert hc["connector"] == "google_docs"

    def test_healthcheck_missing_credentials(self):
        conn = GoogleDocsConnector(folder_ids=[])  # no token, no SA path
        hc = conn.healthcheck()
        assert hc["status"] == "error"
        assert "RAG_GOOGLE_SERVICE_ACCOUNT_JSON" in hc["detail"]

    def test_healthcheck_api_error(self):
        conn = _make_conn()
        with patch.object(conn, "_drive_get", side_effect=RuntimeError("403 Forbidden")):
            hc = conn.healthcheck()
        assert hc["status"] == "error"
        assert "403" in hc["detail"]

    # ------ mime-type filter ------------------------------------------------

    def test_query_filters_to_docs_mime_type(self):
        conn = _make_conn()
        with patch.object(conn, "_drive_get", return_value=_drive_response([])) as mock_dg:
            conn.list_items()
        params = mock_dg.call_args[0][2]
        assert "application/vnd.google-apps.document" in params["q"]
