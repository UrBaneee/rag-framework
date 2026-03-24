"""Tests for SlackConnector — Task 15.3.

All tests are fully offline: _slack_get is monkeypatched to return
canned dicts so no network access or Slack credentials are needed.
"""

from __future__ import annotations

from unittest.mock import patch, call

import pytest

from rag.infra.connectors.slack_connector import (
    SlackConnector,
    _flatten_thread,
    _format_message,
    _stable_source_id,
)


# ---------------------------------------------------------------------------
# Fixtures — canned Slack API payloads
# ---------------------------------------------------------------------------


def _history_response(messages: list[dict]) -> dict:
    return {"ok": True, "messages": messages, "has_more": False}


def _replies_response(messages: list[dict]) -> dict:
    return {"ok": True, "messages": messages, "has_more": False}


def _auth_response(team: str = "MyTeam", user: str = "bot") -> dict:
    return {"ok": True, "team": team, "user": user}


def _msg(ts: str, text: str, user: str = "U001", thread_ts: str | None = None, reply_count: int = 0) -> dict:
    m: dict = {"ts": ts, "text": text, "user": user, "reply_count": reply_count}
    if thread_ts:
        m["thread_ts"] = thread_ts
    return m


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestFormatMessage:
    def test_basic(self):
        line = _format_message({"ts": "1700000000.000100", "user": "U123", "text": "Hello"})
        assert "U123" in line
        assert "Hello" in line
        assert "1700000000.000100" in line

    def test_missing_user_falls_back(self):
        line = _format_message({"ts": "1.0", "text": "hi"})
        assert "unknown" in line

    def test_uses_username_when_no_user(self):
        line = _format_message({"ts": "1.0", "username": "bot_name", "text": "ping"})
        assert "bot_name" in line


class TestFlattenThread:
    def test_root_only(self):
        root = {"ts": "1.0", "user": "U1", "text": "Root message"}
        text = _flatten_thread(root, [])
        assert "Root message" in text
        assert text.count("\n") == 0

    def test_root_and_replies(self):
        root = {"ts": "1.0", "user": "U1", "text": "Root"}
        replies = [
            {"ts": "1.1", "user": "U2", "text": "Reply 1"},
            {"ts": "1.2", "user": "U3", "text": "Reply 2"},
        ]
        text = _flatten_thread(root, replies)
        lines = text.splitlines()
        assert len(lines) == 3
        assert "Root" in lines[0]
        assert "Reply 1" in lines[1]
        assert "Reply 2" in lines[2]

    def test_skips_root_in_replies_list(self):
        """API sometimes includes root message in replies; it must not be duplicated."""
        root = {"ts": "1.0", "user": "U1", "text": "Root"}
        replies = [
            root,  # duplicate root
            {"ts": "1.1", "user": "U2", "text": "Reply"},
        ]
        text = _flatten_thread(root, replies)
        lines = text.splitlines()
        assert len(lines) == 2

    def test_oldest_reply_appears_before_newest(self):
        root = {"ts": "1.0", "user": "U1", "text": "Start"}
        replies = [
            {"ts": "1.1", "user": "U2", "text": "First reply"},
            {"ts": "1.9", "user": "U3", "text": "Last reply"},
        ]
        text = _flatten_thread(root, replies)
        assert text.index("First reply") < text.index("Last reply")


class TestStableSourceId:
    def test_format(self):
        sid = _stable_source_id("C123", "1700000000.000100")
        assert sid == "slack:C123/1700000000.000100"

    def test_deterministic(self):
        assert _stable_source_id("C1", "ts1") == _stable_source_id("C1", "ts1")

    def test_different_channels_different_ids(self):
        assert _stable_source_id("C1", "ts1") != _stable_source_id("C2", "ts1")


# ---------------------------------------------------------------------------
# Integration tests — SlackConnector with mocked _slack_get
# ---------------------------------------------------------------------------


_PATCH = "rag.infra.connectors.slack_connector._slack_get"


class TestSlackConnector:
    def _make_connector(self, channels: list[str] | None = None) -> SlackConnector:
        return SlackConnector(
            token="xoxb-fake",
            channel_ids=channels or ["C001"],
            fetch_replies=True,
        )

    # ------ basic fetch -------------------------------------------------------

    def test_list_items_returns_artifacts(self):
        msgs = [_msg("1700000002.0", "Hello", "U1")]
        with patch(_PATCH, return_value=_history_response(msgs)):
            conn = self._make_connector()
            arts = conn.list_items(since_cursor="")
        assert len(arts) == 1
        assert arts[0].source_type == "slack"
        assert "Hello" in arts[0].content_text

    def test_artifact_metadata_fields(self):
        msgs = [_msg("1700000001.0", "msg", "U2")]
        with patch(_PATCH, return_value=_history_response(msgs)):
            conn = self._make_connector(["C999"])
            arts = conn.list_items()
        a = arts[0]
        assert a.metadata["channel_id"] == "C999"
        assert a.metadata["message_ts"] == "1700000001.0"
        assert a.metadata["user"] == "U2"

    def test_empty_channel_returns_empty(self):
        with patch(_PATCH, return_value=_history_response([])):
            conn = self._make_connector()
            arts = conn.list_items()
        assert arts == []

    # ------ cursor behaviour --------------------------------------------------

    def test_cursor_advances_to_newest_ts(self):
        msgs = [
            _msg("1700000003.0", "newer"),
            _msg("1700000001.0", "older"),
        ]
        with patch(_PATCH, return_value=_history_response(msgs)):
            conn = self._make_connector()
            conn.list_items(since_cursor="")
        assert conn.next_cursor() == "1700000003.0"

    def test_cursor_empty_before_first_call(self):
        conn = self._make_connector()
        assert conn.next_cursor() == ""

    def test_since_cursor_passed_as_oldest_param(self):
        with patch(_PATCH, return_value=_history_response([])) as mock_get:
            conn = self._make_connector()
            conn.list_items(since_cursor="1700000010.0")
        call_kwargs = mock_get.call_args
        params = call_kwargs[0][2]  # positional: method, token, params
        assert params.get("oldest") == "1700000010.0"

    def test_no_oldest_param_when_cursor_empty(self):
        with patch(_PATCH, return_value=_history_response([])) as mock_get:
            conn = self._make_connector()
            conn.list_items(since_cursor="")
        params = mock_get.call_args[0][2]
        assert "oldest" not in params

    # ------ ordering ----------------------------------------------------------

    def test_artifacts_ordered_oldest_first(self):
        # conversations.history returns newest-first; connector reverses
        msgs = [
            _msg("1700000003.0", "newest"),
            _msg("1700000002.0", "middle"),
            _msg("1700000001.0", "oldest"),
        ]
        with patch(_PATCH, return_value=_history_response(msgs)):
            conn = self._make_connector()
            arts = conn.list_items()
        tss = [a.metadata["message_ts"] for a in arts]
        assert tss == sorted(tss)

    # ------ thread replies ----------------------------------------------------

    def test_threaded_message_fetches_replies(self):
        root_msg = _msg("1700000001.0", "Root", reply_count=2)
        root_msg["thread_ts"] = "1700000001.0"
        reply1 = _msg("1700000002.0", "Reply1", thread_ts="1700000001.0")
        reply2 = _msg("1700000003.0", "Reply2", thread_ts="1700000001.0")

        history = _history_response([root_msg])
        replies = _replies_response([root_msg, reply1, reply2])

        call_count = 0

        def fake_get(method, token, params, **kw):
            nonlocal call_count
            call_count += 1
            if method == "conversations.history":
                return history
            if method == "conversations.replies":
                return replies
            return {"ok": True}

        with patch(_PATCH, side_effect=fake_get):
            conn = self._make_connector()
            arts = conn.list_items()

        assert len(arts) == 1
        text = arts[0].content_text
        assert "Root" in text
        assert "Reply1" in text
        assert "Reply2" in text

    def test_reply_messages_not_duplicated_as_standalone(self):
        """Sub-messages (thread_ts != ts) must be skipped at top level."""
        root = _msg("1700000001.0", "Root", reply_count=1)
        root["thread_ts"] = "1700000001.0"
        reply = _msg("1700000002.0", "Reply", thread_ts="1700000001.0")

        history = _history_response([root, reply])

        def fake_get(method, token, params, **kw):
            if method == "conversations.history":
                return history
            return _replies_response([root, reply])

        with patch(_PATCH, side_effect=fake_get):
            conn = self._make_connector()
            arts = conn.list_items()

        # Only one artifact for the thread root
        assert len(arts) == 1

    def test_no_replies_fetched_when_disabled(self):
        root = _msg("1700000001.0", "Root", reply_count=3)
        root["thread_ts"] = "1700000001.0"

        with patch(_PATCH, return_value=_history_response([root])) as mock_get:
            conn = SlackConnector(
                token="xoxb-fake",
                channel_ids=["C001"],
                fetch_replies=False,
            )
            arts = conn.list_items()

        # Should only have called conversations.history, not replies
        called_methods = [c[0][0] for c in mock_get.call_args_list]
        assert "conversations.replies" not in called_methods

    # ------ multi-channel -----------------------------------------------------

    def test_multiple_channels_synced(self):
        with patch(_PATCH, return_value=_history_response([_msg("1.0", "msg")])):
            conn = SlackConnector(
                token="xoxb-fake",
                channel_ids=["C001", "C002"],
                fetch_replies=False,
            )
            arts = conn.list_items()
        assert len(arts) == 2
        channel_ids = {a.metadata["channel_id"] for a in arts}
        assert channel_ids == {"C001", "C002"}

    def test_channel_error_does_not_abort_others(self):
        call_count = 0

        def fake_get(method, token, params, **kw):
            nonlocal call_count
            call_count += 1
            if params.get("channel") == "C001":
                raise RuntimeError("channel not found")
            return _history_response([_msg("1.0", "ok")])

        with patch(_PATCH, side_effect=fake_get):
            conn = SlackConnector(
                token="xoxb-fake",
                channel_ids=["C001", "C002"],
                fetch_replies=False,
            )
            arts = conn.list_items()

        # C002 should still produce an artifact
        assert len(arts) == 1
        assert arts[0].metadata["channel_id"] == "C002"

    # ------ healthcheck -------------------------------------------------------

    def test_healthcheck_ok(self):
        with patch(_PATCH, return_value=_auth_response("Acme", "bot")):
            conn = self._make_connector()
            hc = conn.healthcheck()
        assert hc["status"] == "ok"
        assert "Acme" in hc["detail"]

    def test_healthcheck_missing_token(self):
        conn = SlackConnector(token="", channel_ids=["C1"])
        hc = conn.healthcheck()
        assert hc["status"] == "error"
        assert "RAG_SLACK_BOT_TOKEN" in hc["detail"]

    def test_healthcheck_api_error(self):
        with patch(_PATCH, side_effect=RuntimeError("invalid_auth")):
            conn = self._make_connector()
            hc = conn.healthcheck()
        assert hc["status"] == "error"
        assert "invalid_auth" in hc["detail"]

    # ------ source_id stability -----------------------------------------------

    def test_source_id_stable_across_calls(self):
        msgs = [_msg("1700000001.0", "Hello")]
        with patch(_PATCH, return_value=_history_response(msgs)):
            conn = self._make_connector(["C123"])
            arts1 = conn.list_items()
        with patch(_PATCH, return_value=_history_response(msgs)):
            arts2 = conn.list_items()
        assert arts1[0].source_id == arts2[0].source_id

    def test_external_url_contains_channel_and_ts(self):
        msgs = [_msg("1700000001.000200", "msg")]
        with patch(_PATCH, return_value=_history_response(msgs)):
            conn = self._make_connector(["C999"])
            arts = conn.list_items()
        assert "C999" in arts[0].external_url
        assert "1700000001000200" in arts[0].external_url
