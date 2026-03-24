"""Notion connector — Task 15.4.

Pulls Notion pages (and database row pages) since a saved cursor,
converts their block-rich content into plain-text ``SourceArtifact``
objects, and advances the cursor.

The cursor is an ISO 8601 timestamp string (``last_edited_time`` of the
newest page seen).  An empty cursor means "fetch all pages".

The connector searches for pages via the Notion Search API with a
``last_edited_time`` filter, then fetches each page's block tree to
reconstruct plain text.  Only the standard library (``urllib``) is used;
no ``notion-client`` package is required.

Environment variables for credentials:

    RAG_NOTION_TOKEN         — Notion integration token (``secret_…``)
    RAG_NOTION_DATABASE_IDS  — Comma-separated Notion database UUIDs to
                               sync (optional; empty = search all accessible
                               pages via the integration)

Block types rendered to text:

    paragraph, heading_1/2/3, bulleted_list_item, numbered_list_item,
    toggle, quote, callout, code, to_do, child_page (title only)

All other block types are silently skipped (images, files, embeds, etc.).

Usage::

    import os
    os.environ["RAG_NOTION_TOKEN"] = "secret_..."
    os.environ["RAG_NOTION_DATABASE_IDS"] = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

    connector = NotionConnector()
    artifacts = connector.list_items(since_cursor="")
    cursor = connector.next_cursor()
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from rag.core.contracts.source_artifact import SourceArtifact
from rag.core.interfaces.source_connector import BaseSourceConnector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NOTION_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"
_DEFAULT_PAGE_SIZE = 20

# Block types that carry plain-text content we want to extract
_TEXT_BLOCK_TYPES = {
    "paragraph",
    "heading_1",
    "heading_2",
    "heading_3",
    "bulleted_list_item",
    "numbered_list_item",
    "toggle",
    "quote",
    "callout",
    "code",
    "to_do",
}


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------


def _notion_request(
    method: str,
    path: str,
    token: str,
    body: Optional[dict] = None,
    timeout: int = 15,
) -> dict[str, Any]:
    """Send a request to the Notion API and return the parsed JSON body.

    Args:
        method: HTTP method (``"GET"`` or ``"POST"``).
        path: API path, e.g. ``"/search"`` or ``"/blocks/{id}/children"``.
        token: Notion integration token.
        body: Optional JSON request body (for POST).
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON dict.

    Raises:
        RuntimeError: On HTTP error or non-success response.
    """
    url = f"{_NOTION_API_BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Notion API HTTP {exc.code} for {path}: {body_text}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Notion API request failed for {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Block-to-text helpers
# ---------------------------------------------------------------------------


def _rich_text_to_str(rich_text_list: list[dict[str, Any]]) -> str:
    """Concatenate the plain_text fields from a Notion rich_text array."""
    return "".join(rt.get("plain_text", "") for rt in rich_text_list)


def _block_to_text(block: dict[str, Any]) -> Optional[str]:
    """Convert a single Notion block to a plain-text line.

    Args:
        block: A Notion block object from the blocks API.

    Returns:
        Plain-text string for the block, or ``None`` if the block type is
        not supported / has no text content.
    """
    block_type = block.get("type", "")
    if block_type not in _TEXT_BLOCK_TYPES:
        if block_type == "child_page":
            title = block.get("child_page", {}).get("title", "")
            return f"[Page: {title}]" if title else None
        return None

    type_data = block.get(block_type, {})
    rich_text = type_data.get("rich_text", [])
    text = _rich_text_to_str(rich_text)
    if not text:
        return None

    # Add structural hints for headings
    if block_type == "heading_1":
        return f"# {text}"
    if block_type == "heading_2":
        return f"## {text}"
    if block_type == "heading_3":
        return f"### {text}"
    if block_type == "bulleted_list_item":
        return f"- {text}"
    if block_type == "numbered_list_item":
        return f"1. {text}"
    if block_type == "quote":
        return f"> {text}"
    if block_type == "code":
        lang = type_data.get("language", "")
        return f"```{lang}\n{text}\n```"
    if block_type == "to_do":
        checked = type_data.get("checked", False)
        marker = "[x]" if checked else "[ ]"
        return f"{marker} {text}"
    return text


def blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    """Convert a list of Notion blocks to a plain-text document string.

    Args:
        blocks: List of Notion block objects.

    Returns:
        Multi-line plain-text string.  Empty string if no supported blocks.
    """
    lines: list[str] = []
    for block in blocks:
        line = _block_to_text(block)
        if line:
            lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


def _page_title(page: dict[str, Any]) -> str:
    """Extract the title from a Notion page object."""
    props = page.get("properties", {})
    # Pages have a "title" property; database rows may use a different key
    for prop in props.values():
        if prop.get("type") == "title":
            rich_text = prop.get("title", [])
            title = _rich_text_to_str(rich_text)
            if title:
                return title
    return "(Untitled)"


def _stable_source_id(page_id: str) -> str:
    return f"notion:{page_id}"


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class NotionConnector(BaseSourceConnector):
    """Notion connector — fetches pages and database rows via the Notion API.

    The cursor is the ``last_edited_time`` ISO timestamp of the most recently
    edited page seen.  Pages are filtered to those edited after the cursor.

    Args:
        token: Notion integration token.  Falls back to ``RAG_NOTION_TOKEN``.
        database_ids: List of database UUIDs to query.  Falls back to
            ``RAG_NOTION_DATABASE_IDS`` (comma-separated).  If empty, the
            Search API is used to find all accessible pages.
        page_size: Number of results per API call.
        fetch_blocks: If True, fetch block children to build content_text.
            Set to False in tests that only care about metadata.
    """

    connector_name = "notion"

    def __init__(
        self,
        token: Optional[str] = None,
        database_ids: Optional[list[str]] = None,
        page_size: int = _DEFAULT_PAGE_SIZE,
        fetch_blocks: bool = True,
    ) -> None:
        self._token = token or os.environ.get("RAG_NOTION_TOKEN", "")
        raw_ids = os.environ.get("RAG_NOTION_DATABASE_IDS", "")
        self._database_ids: list[str] = database_ids or (
            [d.strip() for d in raw_ids.split(",") if d.strip()] if raw_ids else []
        )
        self._page_size = page_size
        self._fetch_blocks = fetch_blocks
        self._cursor: str = ""

    # ------------------------------------------------------------------
    # BaseSourceConnector implementation
    # ------------------------------------------------------------------

    def list_items(self, since_cursor: str = "") -> list[SourceArtifact]:
        """Fetch pages edited after ``since_cursor``.

        Args:
            since_cursor: ISO 8601 timestamp.  Pages with
                ``last_edited_time > since_cursor`` are returned.
                Empty = all accessible pages.

        Returns:
            List of ``SourceArtifact`` objects, one per page.
        """
        pages = self._search_pages(since_cursor)
        artifacts: list[SourceArtifact] = []
        max_ts = since_cursor

        for page in pages:
            try:
                artifact = self._page_to_artifact(page)
                if artifact:
                    artifacts.append(artifact)
                    ts = page.get("last_edited_time", "")
                    if ts > max_ts:
                        max_ts = ts
            except Exception as exc:
                logger.warning("Failed to convert page %s: %s", page.get("id"), exc)

        self._cursor = max_ts
        return artifacts

    def next_cursor(self) -> str:
        """Return the ``last_edited_time`` of the newest page seen."""
        return self._cursor

    def healthcheck(self) -> dict:
        """Call the Notion Users API to verify the token."""
        if not self._token:
            return {
                "status": "error",
                "connector": self.connector_name,
                "detail": "RAG_NOTION_TOKEN is not configured.",
            }
        try:
            data = _notion_request("GET", "/users/me", self._token)
            name = data.get("name") or data.get("bot", {}).get("owner", {}).get("type", "unknown")
            return {
                "status": "ok",
                "connector": self.connector_name,
                "detail": f"Authenticated as {name}",
            }
        except Exception as exc:
            return {
                "status": "error",
                "connector": self.connector_name,
                "detail": str(exc),
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _search_pages(self, since_cursor: str) -> list[dict[str, Any]]:
        """Return pages from databases (or Search API) edited after cursor."""
        if self._database_ids:
            pages: list[dict[str, Any]] = []
            for db_id in self._database_ids:
                try:
                    pages.extend(self._query_database(db_id, since_cursor))
                except Exception as exc:
                    logger.warning("Failed to query database %s: %s", db_id, exc)
            return pages

        # Fallback: Search API for all accessible pages
        return self._search_all_pages(since_cursor)

    def _query_database(
        self, database_id: str, since_cursor: str
    ) -> list[dict[str, Any]]:
        """Query a Notion database for pages edited after the cursor."""
        body: dict[str, Any] = {"page_size": self._page_size}
        if since_cursor:
            body["filter"] = {
                "timestamp": "last_edited_time",
                "last_edited_time": {"after": since_cursor},
            }

        data = _notion_request(
            "POST",
            f"/databases/{database_id}/query",
            self._token,
            body=body,
        )
        return data.get("results", [])

    def _search_all_pages(self, since_cursor: str) -> list[dict[str, Any]]:
        """Use the Notion Search API to find all accessible pages."""
        body: dict[str, Any] = {
            "filter": {"value": "page", "property": "object"},
            "page_size": self._page_size,
            "sort": {
                "direction": "ascending",
                "timestamp": "last_edited_time",
            },
        }
        if since_cursor:
            body["filter_properties"] = {"last_edited_time": {"after": since_cursor}}

        data = _notion_request("POST", "/search", self._token, body=body)
        return data.get("results", [])

    def _fetch_block_children(self, block_id: str) -> list[dict[str, Any]]:
        """Fetch all children of a block (page or block), with pagination."""
        blocks: list[dict[str, Any]] = []
        start_cursor: Optional[str] = None

        while True:
            params = f"?page_size=100"
            if start_cursor:
                params += f"&start_cursor={start_cursor}"
            data = _notion_request(
                "GET",
                f"/blocks/{block_id}/children{params}",
                self._token,
            )
            blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")
            if not start_cursor:
                break

        return blocks

    def _page_to_artifact(self, page: dict[str, Any]) -> Optional[SourceArtifact]:
        """Convert a Notion page object to a ``SourceArtifact``."""
        page_id = page.get("id", "")
        if not page_id:
            return None

        title = _page_title(page)
        last_edited = page.get("last_edited_time", "")
        created = page.get("created_time", "")
        url = page.get("url", "")
        workspace = page.get("workspace_name", "")
        parent = page.get("parent", {})
        parent_type = parent.get("type", "")
        parent_id = parent.get(parent_type, "") if parent_type else ""

        content_text = title
        if self._fetch_blocks:
            try:
                blocks = self._fetch_block_children(page_id)
                body_text = blocks_to_text(blocks)
                if body_text:
                    content_text = f"{title}\n\n{body_text}"
            except Exception as exc:
                logger.warning("Failed to fetch blocks for page %s: %s", page_id, exc)

        return SourceArtifact(
            source_type="notion",
            source_id=_stable_source_id(page_id),
            external_url=url,
            content_text=content_text,
            mime_type="text/plain",
            metadata={
                "page_id": page_id,
                "title": title,
                "last_edited_time": last_edited,
                "created_time": created,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "workspace": workspace,
            },
            cursor_after=last_edited,
        )
