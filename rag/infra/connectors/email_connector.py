"""Email connector — Task 15.2.

Pulls email messages from an IMAP mailbox since a saved cursor (UID),
converts each message into a ``SourceArtifact``, and advances the cursor.

The cursor is the highest seen IMAP UID as a decimal string.  An empty
cursor means "start from UID 1" (fetch all).

Credentials are read from environment variables so they are never stored
in config files:

    RAG_EMAIL_SERVER   — IMAP hostname (e.g. ``imap.gmail.com``)
    RAG_EMAIL_PORT     — IMAP port; defaults to 993 (IMAP over SSL)
    RAG_EMAIL_USER     — mailbox username / email address
    RAG_EMAIL_PASSWORD — mailbox password or app-specific password
    RAG_EMAIL_MAILBOX  — mailbox folder to sync; defaults to ``INBOX``

The connector uses only the Python standard library (``imaplib``, ``email``)
so no extra dependencies are required.

Usage::

    import os
    os.environ["RAG_EMAIL_SERVER"] = "imap.example.com"
    os.environ["RAG_EMAIL_USER"] = "user@example.com"
    os.environ["RAG_EMAIL_PASSWORD"] = "secret"

    connector = EmailConnector()
    artifacts = connector.list_items(since_cursor="")
    cursor = connector.next_cursor()
    # persist cursor via DocStore.save_connector_cursor("email", cursor)
"""

from __future__ import annotations

import email
import email.header
import email.policy
import hashlib
import imaplib
import logging
import os
from email.message import Message
from typing import Optional

from rag.core.contracts.source_artifact import SourceArtifact
from rag.core.interfaces.source_connector import BaseSourceConnector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PORT = 993
_DEFAULT_MAILBOX = "INBOX"
_MAX_ATTACHMENT_BYTES = 5 * 1024 * 1024  # 5 MB per attachment

# MIME types we will extract as text content
_TEXT_MIME_TYPES = {"text/plain", "text/html"}

# MIME types we forward as binary (e.g. PDF for OCR pipeline)
_BINARY_MIME_TYPES = {"application/pdf"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_header_value(raw: str) -> str:
    """Decode a potentially RFC 2047-encoded header value to a plain string."""
    parts = email.header.decode_header(raw or "")
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return " ".join(decoded).strip()


def _stable_source_id(message_id: str, uid: str) -> str:
    """Return a stable source_id from the Message-ID header.

    Falls back to a SHA-256 of the IMAP UID if Message-ID is absent.
    """
    if message_id:
        # Strip angle brackets and whitespace
        clean = message_id.strip().strip("<>")
        return f"email:{clean}"
    # Fallback: hash the UID so the ID is still deterministic per server
    uid_hash = hashlib.sha256(uid.encode()).hexdigest()[:16]
    return f"email:uid-{uid_hash}"


def _extract_text(msg: Message) -> str:
    """Extract the best plain-text representation of an email message.

    Prefers ``text/plain`` parts; falls back to ``text/html`` if no plain
    part is found.  HTML tags are stripped with a simple approach (no
    external dependency).
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain" and not part.get_filename():
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload:
                    plain_parts.append(payload.decode(charset, errors="replace"))
            elif ct == "text/html" and not part.get_filename():
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                if payload:
                    html_parts.append(payload.decode(charset, errors="replace"))
    else:
        ct = msg.get_content_type()
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset() or "utf-8"
        if payload:
            text = payload.decode(charset, errors="replace")
            if ct == "text/plain":
                plain_parts.append(text)
            elif ct == "text/html":
                html_parts.append(text)

    if plain_parts:
        return "\n\n".join(plain_parts).strip()

    # Minimal HTML tag stripper (no regex/beautifulsoup dependency)
    if html_parts:
        raw_html = "\n\n".join(html_parts)
        result: list[str] = []
        in_tag = False
        for ch in raw_html:
            if ch == "<":
                in_tag = True
            elif ch == ">":
                in_tag = False
            elif not in_tag:
                result.append(ch)
        return "".join(result).strip()

    return ""


def _extract_attachments(msg: Message) -> list[tuple[str, bytes, str]]:
    """Return (filename, content_bytes, mime_type) for interesting attachments.

    Only returns attachments whose MIME type is in ``_BINARY_MIME_TYPES`` and
    whose size is within ``_MAX_ATTACHMENT_BYTES``.
    """
    attachments: list[tuple[str, bytes, str]] = []
    if not msg.is_multipart():
        return attachments

    for part in msg.walk():
        filename = part.get_filename()
        if not filename:
            continue
        ct = part.get_content_type()
        if ct not in _BINARY_MIME_TYPES:
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if len(payload) > _MAX_ATTACHMENT_BYTES:
            logger.debug(
                "Skipping attachment %s (%.1f MB > limit)", filename, len(payload) / 1e6
            )
            continue
        attachments.append((filename, payload, ct))

    return attachments


def _msg_to_artifacts(
    uid: str,
    raw_bytes: bytes,
    external_url_template: str = "",
) -> list[SourceArtifact]:
    """Parse raw RFC 2822 bytes and return one or more ``SourceArtifact`` objects.

    Always produces an artifact for the message body.  Produces additional
    artifacts for qualifying binary attachments (e.g. PDFs).

    Args:
        uid: IMAP UID string.
        raw_bytes: Full RFC 2822 message bytes.
        external_url_template: Optional format string with ``{message_id}``
            placeholder, e.g. a web-mail URL.

    Returns:
        List of ``SourceArtifact`` — at least one (the body).
    """
    msg = email.message_from_bytes(raw_bytes, policy=email.policy.compat32)

    message_id = _decode_header_value(msg.get("Message-ID", ""))
    subject = _decode_header_value(msg.get("Subject", "(no subject)"))
    sender = _decode_header_value(msg.get("From", ""))
    date_str = msg.get("Date", "")

    source_id = _stable_source_id(message_id, uid)
    external_url = external_url_template.format(message_id=message_id) if external_url_template else ""

    body_text = _extract_text(msg)

    artifacts: list[SourceArtifact] = []

    # Body artifact
    artifacts.append(
        SourceArtifact(
            source_type="email",
            source_id=source_id,
            external_url=external_url,
            content_text=body_text,
            mime_type="text/plain",
            metadata={
                "subject": subject,
                "from": sender,
                "date": date_str,
                "message_id": message_id,
                "uid": uid,
            },
            cursor_after=uid,
        )
    )

    # Attachment artifacts (e.g. PDF)
    for filename, content_bytes, ct in _extract_attachments(msg):
        attach_id = f"{source_id}/attachment/{filename}"
        artifacts.append(
            SourceArtifact(
                source_type="email",
                source_id=attach_id,
                external_url=external_url,
                content_bytes=content_bytes,
                mime_type=ct,
                metadata={
                    "subject": subject,
                    "from": sender,
                    "date": date_str,
                    "message_id": message_id,
                    "uid": uid,
                    "attachment_filename": filename,
                },
                cursor_after=uid,
            )
        )

    return artifacts


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class EmailConnector(BaseSourceConnector):
    """IMAP email connector.

    Fetches messages by UID range since the last stored cursor.  Each
    call to ``list_items`` opens a new IMAP connection (stateless between
    calls) and updates the internal cursor to the highest seen UID.

    Args:
        server: IMAP hostname.  Falls back to ``RAG_EMAIL_SERVER`` env var.
        port: IMAP SSL port.  Falls back to ``RAG_EMAIL_PORT`` or 993.
        user: Mailbox username.  Falls back to ``RAG_EMAIL_USER``.
        password: Mailbox password.  Falls back to ``RAG_EMAIL_PASSWORD``.
        mailbox: Folder to sync.  Falls back to ``RAG_EMAIL_MAILBOX`` or ``INBOX``.
        batch_size: Maximum number of messages to fetch per call.
        external_url_template: Optional URL template with ``{message_id}`` placeholder.
    """

    connector_name = "email"

    def __init__(
        self,
        server: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        mailbox: Optional[str] = None,
        batch_size: int = 50,
        external_url_template: str = "",
    ) -> None:
        self._server = server or os.environ.get("RAG_EMAIL_SERVER", "")
        self._port = port or int(os.environ.get("RAG_EMAIL_PORT", str(_DEFAULT_PORT)))
        self._user = user or os.environ.get("RAG_EMAIL_USER", "")
        self._password = password or os.environ.get("RAG_EMAIL_PASSWORD", "")
        self._mailbox = mailbox or os.environ.get("RAG_EMAIL_MAILBOX", _DEFAULT_MAILBOX)
        self._batch_size = batch_size
        self._external_url_template = external_url_template
        self._cursor: str = ""

    # ------------------------------------------------------------------
    # BaseSourceConnector implementation
    # ------------------------------------------------------------------

    def list_items(self, since_cursor: str = "") -> list[SourceArtifact]:
        """Fetch email messages since ``since_cursor`` (IMAP UID).

        Args:
            since_cursor: Decimal IMAP UID string.  Fetches UIDs > this value.
                Empty string means fetch from UID 1.

        Returns:
            List of ``SourceArtifact`` objects (body + attachment artifacts).
        """
        since_uid = int(since_cursor) if since_cursor.strip().isdigit() else 0
        search_criterion = f"UID {since_uid + 1}:*"

        try:
            with imaplib.IMAP4_SSL(self._server, self._port) as imap:
                imap.login(self._user, self._password)
                imap.select(self._mailbox, readonly=True)

                _, uid_data = imap.uid("search", None, search_criterion)
                if not uid_data or not uid_data[0]:
                    self._cursor = since_cursor
                    return []

                uid_list: list[str] = uid_data[0].decode().split()
                if not uid_list:
                    self._cursor = since_cursor
                    return []

                # Limit batch size
                uid_list = uid_list[: self._batch_size]

                artifacts: list[SourceArtifact] = []
                max_uid = since_uid

                for uid in uid_list:
                    try:
                        _, msg_data = imap.uid("fetch", uid, "(RFC822)")
                        if not msg_data or not msg_data[0]:
                            continue
                        raw = msg_data[0][1]
                        if not isinstance(raw, bytes):
                            continue
                        for artifact in _msg_to_artifacts(
                            uid=uid,
                            raw_bytes=raw,
                            external_url_template=self._external_url_template,
                        ):
                            artifacts.append(artifact)
                        uid_int = int(uid)
                        if uid_int > max_uid:
                            max_uid = uid_int
                    except Exception as exc:
                        logger.warning("Failed to fetch UID %s: %s", uid, exc)

                self._cursor = str(max_uid) if max_uid > since_uid else since_cursor
                return artifacts

        except imaplib.IMAP4.error as exc:
            logger.error("IMAP error in list_items: %s", exc)
            self._cursor = since_cursor
            return []

    def next_cursor(self) -> str:
        """Return the highest IMAP UID seen in the last ``list_items`` call."""
        return self._cursor

    def healthcheck(self) -> dict:
        """Probe the IMAP server and return a status dict."""
        if not self._server:
            return {
                "status": "error",
                "connector": self.connector_name,
                "detail": "RAG_EMAIL_SERVER is not configured.",
            }
        try:
            with imaplib.IMAP4_SSL(self._server, self._port) as imap:
                imap.login(self._user, self._password)
                status, _ = imap.select(self._mailbox, readonly=True)
                if status != "OK":
                    return {
                        "status": "degraded",
                        "connector": self.connector_name,
                        "detail": f"Could not select mailbox '{self._mailbox}'.",
                    }
            return {"status": "ok", "connector": self.connector_name, "detail": ""}
        except Exception as exc:
            return {
                "status": "error",
                "connector": self.connector_name,
                "detail": str(exc),
            }
