"""Slack connector — Task 15.3.

Pulls messages from one or more Slack channels since a saved cursor
(latest timestamp), converts each message (and its thread replies) into
a ``SourceArtifact``, and advances the cursor.

The cursor is the Slack ``ts`` (timestamp string) of the newest message
seen.  An empty cursor means "fetch from the beginning of the channel".

Authentication uses a Bot Token (``xoxb-…``) read from the environment:

    RAG_SLACK_BOT_TOKEN   — Slack Bot OAuth token
    RAG_SLACK_CHANNEL_IDS — Comma-separated list of channel IDs to sync
                            (e.g. ``C01234567,C09876543``)

The connector calls the Slack Web API over HTTPS using only ``urllib``
from the standard library so no ``slack_sdk`` dependency is required.
Each channel is synced independently and a per-channel cursor is stored
in the ``cursor_after`` field of each artifact so callers can persist
progress per channel.

One ``SourceArtifact`` is produced per *thread* (root message + replies
flattened into a single ``content_text`` block).  Stand-alone messages
(no replies) also produce one artifact each.

Usage::

    import os
    os.environ["RAG_SLACK_BOT_TOKEN"] = "xoxb-..."
    os.environ["RAG_SLACK_CHANNEL_IDS"] = "C01234567"

    connector = SlackConnector()
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

_SLACK_API_BASE = "https://slack.com/api"
_DEFAULT_BATCH_SIZE = 200


# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------


def _slack_get(
    method: str,
    token: str,
    params: dict[str, str],
    timeout: int = 15,
) -> dict[str, Any]:
    """Call a Slack Web API method (GET) and return the parsed JSON response.

    Args:
        method: Slack API method name, e.g. ``"conversations.history"``.
        token: Bot token for the ``Authorization`` header.
        params: Query-string parameters.
        timeout: Socket timeout in seconds.

    Returns:
        Parsed JSON dict.

    Raises:
        RuntimeError: On HTTP error or non-ok Slack response.
    """
    url = f"{_SLACK_API_BASE}/{method}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Slack API request failed: {exc}") from exc

    data: dict[str, Any] = json.loads(body)
    if not data.get("ok"):
        raise RuntimeError(f"Slack API error: {data.get('error', 'unknown')}")
    return data


# ---------------------------------------------------------------------------
# Message flattening helpers
# ---------------------------------------------------------------------------


def _format_message(msg: dict[str, Any]) -> str:
    """Return a formatted line for a single Slack message dict."""
    user = msg.get("user") or msg.get("username") or "unknown"
    ts = msg.get("ts", "")
    text = msg.get("text", "").strip()
    return f"[{ts}] <{user}>: {text}"


def _flatten_thread(
    root: dict[str, Any],
    replies: list[dict[str, Any]],
) -> str:
    """Flatten a root message and its replies into a single text block.

    The root message appears first, followed by replies in chronological
    order (oldest first).
    """
    lines = [_format_message(root)]
    for reply in replies:
        # Skip the root message if the API included it in the reply list
        if reply.get("ts") == root.get("ts"):
            continue
        lines.append(_format_message(reply))
    return "\n".join(lines)


def _stable_source_id(channel_id: str, thread_ts: str) -> str:
    """Return a stable source_id for a thread or standalone message."""
    return f"slack:{channel_id}/{thread_ts}"


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class SlackConnector(BaseSourceConnector):
    """Slack connector — pulls channel messages and threads via the Web API.

    Each channel is synced independently.  The cursor is the ``ts`` of the
    most recent message seen across all configured channels, stored as a
    plain string.  Per-channel cursors are embedded in each artifact's
    ``cursor_after`` field so callers can do finer-grained checkpointing.

    Args:
        token: Slack Bot token.  Falls back to ``RAG_SLACK_BOT_TOKEN``.
        channel_ids: List of channel IDs to sync.  Falls back to
            ``RAG_SLACK_CHANNEL_IDS`` (comma-separated).
        batch_size: Max messages per channel per call (``limit`` param).
        fetch_replies: If True, fetch thread replies for threaded messages.
    """

    connector_name = "slack"

    def __init__(
        self,
        token: Optional[str] = None,
        channel_ids: Optional[list[str]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        fetch_replies: bool = True,
    ) -> None:
        self._token = token or os.environ.get("RAG_SLACK_BOT_TOKEN", "")
        raw_ids = os.environ.get("RAG_SLACK_CHANNEL_IDS", "")
        self._channel_ids: list[str] = channel_ids or (
            [c.strip() for c in raw_ids.split(",") if c.strip()] if raw_ids else []
        )
        self._batch_size = batch_size
        self._fetch_replies = fetch_replies
        self._cursor: str = ""

    # ------------------------------------------------------------------
    # BaseSourceConnector implementation
    # ------------------------------------------------------------------

    def list_items(self, since_cursor: str = "") -> list[SourceArtifact]:
        """Fetch messages from all configured channels since ``since_cursor``.

        Args:
            since_cursor: Slack ``ts`` timestamp string.  Messages with
                ``ts > since_cursor`` are returned.  Empty = all messages.

        Returns:
            Ordered list of ``SourceArtifact`` objects (oldest first within
            each channel).
        """
        all_artifacts: list[SourceArtifact] = []
        max_ts = since_cursor

        for channel_id in self._channel_ids:
            try:
                artifacts = self._sync_channel(channel_id, since_cursor)
                all_artifacts.extend(artifacts)
                # Track the highest ts seen across all channels
                for art in artifacts:
                    if art.cursor_after > max_ts:
                        max_ts = art.cursor_after
            except Exception as exc:
                logger.warning("Failed to sync channel %s: %s", channel_id, exc)

        self._cursor = max_ts
        return all_artifacts

    def next_cursor(self) -> str:
        """Return the highest ``ts`` seen in the last ``list_items`` call."""
        return self._cursor

    def healthcheck(self) -> dict:
        """Call ``auth.test`` to verify the token and return a status dict."""
        if not self._token:
            return {
                "status": "error",
                "connector": self.connector_name,
                "detail": "RAG_SLACK_BOT_TOKEN is not configured.",
            }
        try:
            data = _slack_get("auth.test", self._token, {})
            team = data.get("team", "")
            user = data.get("user", "")
            return {
                "status": "ok",
                "connector": self.connector_name,
                "detail": f"Authenticated as {user} on {team}",
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

    def _sync_channel(
        self, channel_id: str, since_cursor: str
    ) -> list[SourceArtifact]:
        """Fetch and convert messages from a single channel."""
        params: dict[str, str] = {
            "channel": channel_id,
            "limit": str(self._batch_size),
            "inclusive": "false",
        }
        if since_cursor:
            params["oldest"] = since_cursor

        data = _slack_get("conversations.history", self._token, params)
        messages: list[dict[str, Any]] = data.get("messages", [])

        # conversations.history returns newest-first; reverse to oldest-first
        messages = list(reversed(messages))

        artifacts: list[SourceArtifact] = []
        for msg in messages:
            try:
                artifact = self._message_to_artifact(channel_id, msg)
                if artifact is not None:
                    artifacts.append(artifact)
            except Exception as exc:
                logger.warning(
                    "Failed to convert message ts=%s in channel %s: %s",
                    msg.get("ts"),
                    channel_id,
                    exc,
                )

        return artifacts

    def _message_to_artifact(
        self, channel_id: str, msg: dict[str, Any]
    ) -> Optional[SourceArtifact]:
        """Convert a single Slack message (with optional thread) to an artifact."""
        ts = msg.get("ts", "")
        if not ts:
            return None

        thread_ts = msg.get("thread_ts", ts)
        reply_count = msg.get("reply_count", 0)

        # Only process root messages (thread_ts == ts) or standalone messages.
        # Sub-messages (replies) are handled as part of their thread root.
        if msg.get("thread_ts") and msg.get("thread_ts") != ts:
            return None  # skip: reply processed with its thread root

        replies: list[dict[str, Any]] = []
        if self._fetch_replies and reply_count > 0:
            replies = self._fetch_thread_replies(channel_id, thread_ts)

        content_text = _flatten_thread(msg, replies)

        return SourceArtifact(
            source_type="slack",
            source_id=_stable_source_id(channel_id, thread_ts),
            external_url=f"https://slack.com/archives/{channel_id}/p{ts.replace('.', '')}",
            content_text=content_text,
            mime_type="text/plain",
            metadata={
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "message_ts": ts,
                "user": msg.get("user") or msg.get("username") or "",
                "reply_count": reply_count,
            },
            cursor_after=ts,
        )

    def _fetch_thread_replies(
        self, channel_id: str, thread_ts: str
    ) -> list[dict[str, Any]]:
        """Fetch thread replies for a given thread_ts."""
        data = _slack_get(
            "conversations.replies",
            self._token,
            {"channel": channel_id, "ts": thread_ts, "limit": "200"},
        )
        return data.get("messages", [])
