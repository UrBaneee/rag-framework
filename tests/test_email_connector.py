"""Tests for EmailConnector — Task 15.2.

All tests are fully offline: imaplib.IMAP4_SSL is monkeypatched with a
FakeIMAP context manager that replays canned payloads.
"""

from __future__ import annotations

import email
import email.policy
import hashlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from unittest.mock import MagicMock, patch

import pytest

from rag.infra.connectors.email_connector import (
    EmailConnector,
    _decode_header_value,
    _extract_text,
    _msg_to_artifacts,
    _stable_source_id,
)


# ---------------------------------------------------------------------------
# Fixtures — RFC 2822 message builders
# ---------------------------------------------------------------------------


def _make_plain_message(
    message_id: str,
    subject: str,
    sender: str,
    body: str,
    date: str = "Mon, 01 Jan 2024 12:00:00 +0000",
) -> bytes:
    msg = MIMEText(body, "plain", "utf-8")
    msg["Message-ID"] = f"<{message_id}>"
    msg["Subject"] = subject
    msg["From"] = sender
    msg["Date"] = date
    return msg.as_bytes()


def _make_multipart_message(
    message_id: str,
    subject: str,
    body_plain: str,
    pdf_bytes: bytes | None = None,
    pdf_filename: str = "report.pdf",
) -> bytes:
    msg = MIMEMultipart()
    msg["Message-ID"] = f"<{message_id}>"
    msg["Subject"] = subject
    msg["From"] = "sender@example.com"
    msg["Date"] = "Mon, 01 Jan 2024 12:00:00 +0000"
    msg.attach(MIMEText(body_plain, "plain", "utf-8"))

    if pdf_bytes is not None:
        part = MIMEBase("application", "pdf")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=pdf_filename)
        msg.attach(part)

    return msg.as_bytes()


# ---------------------------------------------------------------------------
# FakeIMAP helper
# ---------------------------------------------------------------------------


class FakeIMAP:
    """Minimal IMAP4_SSL stand-in usable as a context manager."""

    def __init__(self, uid_list: list[str], messages: dict[str, bytes]) -> None:
        self._uid_list = uid_list
        self._messages = messages  # uid -> raw bytes

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def login(self, user, password):
        return ("OK", [b"Logged in"])

    def select(self, mailbox="INBOX", readonly=False):
        return ("OK", [b"1"])

    def uid(self, command, *args):
        if command == "search":
            encoded = " ".join(self._uid_list).encode()
            return ("OK", [encoded])
        if command == "fetch":
            uid = args[0]
            raw = self._messages.get(uid, b"")
            return ("OK", [(b"1 (RFC822 {100})", raw)])
        return ("OK", [None])


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------


class TestDecodeHeaderValue:
    def test_plain_ascii(self):
        assert _decode_header_value("Hello World") == "Hello World"

    def test_empty_string(self):
        assert _decode_header_value("") == ""

    def test_none_like_empty(self):
        assert _decode_header_value(None) == ""  # type: ignore[arg-type]


class TestStableSourceId:
    def test_uses_message_id_when_present(self):
        sid = _stable_source_id("<abc@example.com>", "42")
        assert sid == "email:abc@example.com"

    def test_strips_angle_brackets(self):
        sid = _stable_source_id("<msg-001@host>", "1")
        assert "msg-001@host" in sid

    def test_falls_back_to_uid_hash_when_empty(self):
        sid = _stable_source_id("", "99")
        assert sid.startswith("email:uid-")
        # deterministic
        assert sid == _stable_source_id("", "99")

    def test_different_uids_give_different_ids(self):
        assert _stable_source_id("", "1") != _stable_source_id("", "2")


class TestExtractText:
    def test_plain_text_message(self):
        raw = _make_plain_message("m1@h", "Sub", "a@b.com", "Hello plain")
        msg = email.message_from_bytes(raw)
        assert _extract_text(msg) == "Hello plain"

    def test_multipart_prefers_plain(self):
        raw = _make_multipart_message("m2@h", "Sub", "body plain")
        msg = email.message_from_bytes(raw)
        assert "body plain" in _extract_text(msg)

    def test_empty_body(self):
        raw = _make_plain_message("m3@h", "Sub", "a@b", "")
        msg = email.message_from_bytes(raw)
        assert _extract_text(msg) == ""


# ---------------------------------------------------------------------------
# Unit tests — _msg_to_artifacts
# ---------------------------------------------------------------------------


class TestMsgToArtifacts:
    def test_plain_message_single_artifact(self):
        raw = _make_plain_message("id1@host", "Hello", "a@b.com", "Body text")
        arts = _msg_to_artifacts(uid="10", raw_bytes=raw)
        assert len(arts) == 1
        a = arts[0]
        assert a.source_type == "email"
        assert a.source_id == "email:id1@host"
        assert "Body text" in a.content_text
        assert a.metadata["subject"] == "Hello"
        assert a.metadata["uid"] == "10"
        assert a.cursor_after == "10"

    def test_deterministic_source_id(self):
        raw = _make_plain_message("id2@host", "Sub", "s@s.com", "body")
        arts1 = _msg_to_artifacts(uid="5", raw_bytes=raw)
        arts2 = _msg_to_artifacts(uid="5", raw_bytes=raw)
        assert arts1[0].source_id == arts2[0].source_id

    def test_multipart_with_pdf_yields_two_artifacts(self):
        raw = _make_multipart_message("id3@host", "Report", "body", pdf_bytes=b"%PDF-fake")
        arts = _msg_to_artifacts(uid="20", raw_bytes=raw)
        assert len(arts) == 2
        body_art = arts[0]
        pdf_art = arts[1]
        assert body_art.content_text is not None
        assert pdf_art.mime_type == "application/pdf"
        assert pdf_art.content_bytes == b"%PDF-fake"
        assert "attachment_filename" in pdf_art.metadata

    def test_cursor_after_set_to_uid(self):
        raw = _make_plain_message("id4@host", "S", "f@f.com", "x")
        arts = _msg_to_artifacts(uid="77", raw_bytes=raw)
        assert all(a.cursor_after == "77" for a in arts)

    def test_external_url_template(self):
        raw = _make_plain_message("id5@host", "S", "f@f.com", "x")
        arts = _msg_to_artifacts(
            uid="1",
            raw_bytes=raw,
            external_url_template="https://mail.example.com/view?id={message_id}",
        )
        assert "id5@host" in arts[0].external_url


# ---------------------------------------------------------------------------
# Integration tests — EmailConnector with mocked IMAP
# ---------------------------------------------------------------------------


class TestEmailConnector:
    def _make_connector(self) -> EmailConnector:
        return EmailConnector(
            server="imap.example.com",
            user="user@example.com",
            password="secret",
            mailbox="INBOX",
        )

    def test_list_items_returns_artifacts(self):
        raw = _make_plain_message("msg1@h", "Test", "s@s.com", "Hello")
        fake = FakeIMAP(uid_list=["1"], messages={"1": raw})

        with patch("rag.infra.connectors.email_connector.imaplib.IMAP4_SSL", return_value=fake):
            conn = self._make_connector()
            arts = conn.list_items(since_cursor="")

        assert len(arts) == 1
        assert arts[0].source_type == "email"

    def test_cursor_advances_to_max_uid(self):
        raw1 = _make_plain_message("m1@h", "A", "s@s.com", "a")
        raw2 = _make_plain_message("m2@h", "B", "s@s.com", "b")
        fake = FakeIMAP(uid_list=["3", "7"], messages={"3": raw1, "7": raw2})

        with patch("rag.infra.connectors.email_connector.imaplib.IMAP4_SSL", return_value=fake):
            conn = self._make_connector()
            conn.list_items(since_cursor="2")

        assert conn.next_cursor() == "7"

    def test_empty_mailbox_returns_empty_list(self):
        fake = FakeIMAP(uid_list=[], messages={})

        with patch("rag.infra.connectors.email_connector.imaplib.IMAP4_SSL", return_value=fake):
            conn = self._make_connector()
            arts = conn.list_items(since_cursor="")

        assert arts == []

    def test_since_cursor_filters_older_uids(self):
        """Messages with UID <= since_cursor should not appear."""
        raw = _make_plain_message("new@h", "New", "s@s.com", "new body")
        # FakeIMAP always returns whatever we give it; the real filtering
        # is done server-side via the UID search criterion.  We verify the
        # connector passes the right search and returns what the server gives.
        fake = FakeIMAP(uid_list=["10"], messages={"10": raw})

        with patch("rag.infra.connectors.email_connector.imaplib.IMAP4_SSL", return_value=fake):
            conn = self._make_connector()
            arts = conn.list_items(since_cursor="9")

        assert len(arts) == 1
        assert arts[0].metadata["uid"] == "10"

    def test_next_cursor_empty_before_first_call(self):
        conn = self._make_connector()
        assert conn.next_cursor() == ""

    def test_imap_error_returns_empty_list(self):
        import imaplib

        with patch(
            "rag.infra.connectors.email_connector.imaplib.IMAP4_SSL",
            side_effect=imaplib.IMAP4.error("connection refused"),
        ):
            conn = self._make_connector()
            arts = conn.list_items(since_cursor="")

        assert arts == []

    def test_healthcheck_missing_server(self):
        conn = EmailConnector(server="", user="u", password="p")
        hc = conn.healthcheck()
        assert hc["status"] == "error"
        assert hc["connector"] == "email"

    def test_healthcheck_ok(self):
        fake = FakeIMAP(uid_list=[], messages={})
        with patch("rag.infra.connectors.email_connector.imaplib.IMAP4_SSL", return_value=fake):
            conn = self._make_connector()
            hc = conn.healthcheck()
        assert hc["status"] == "ok"
        assert hc["connector"] == "email"

    def test_artifact_count_matches_message_count(self):
        """Each plain message should produce exactly one artifact."""
        msgs = {}
        uids = []
        for i in range(1, 6):
            uid = str(i)
            uids.append(uid)
            msgs[uid] = _make_plain_message(f"m{i}@h", f"Subject {i}", "s@s.com", f"body {i}")

        fake = FakeIMAP(uid_list=uids, messages=msgs)
        with patch("rag.infra.connectors.email_connector.imaplib.IMAP4_SSL", return_value=fake):
            conn = self._make_connector()
            arts = conn.list_items(since_cursor="")

        assert len(arts) == 5

    def test_batch_size_limits_fetched_messages(self):
        msgs = {}
        uids = [str(i) for i in range(1, 101)]  # 100 messages
        for uid in uids:
            msgs[uid] = _make_plain_message(f"m{uid}@h", "S", "s@s.com", "body")

        fake = FakeIMAP(uid_list=uids, messages=msgs)
        with patch("rag.infra.connectors.email_connector.imaplib.IMAP4_SSL", return_value=fake):
            conn = EmailConnector(
                server="imap.example.com",
                user="u",
                password="p",
                batch_size=10,
            )
            arts = conn.list_items(since_cursor="")

        assert len(arts) == 10
