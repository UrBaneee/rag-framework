"""SourceArtifact contract — Task 15.1.

Represents a single item fetched from an external connector (email message,
Slack thread, Notion page, Google Doc, …) before it enters the ingest
pipeline.

The artifact carries either raw bytes (``content_bytes``) or decoded text
(``content_text``).  The ingest pipeline will pick whichever is set; if
both are set, ``content_text`` takes precedence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceArtifact:
    """A single item fetched from an external connector.

    Attributes:
        source_type: Connector family identifier, e.g. ``"email"``,
            ``"slack"``, ``"notion"``, ``"google_docs"``.
        source_id: Stable, provider-unique identifier for this item
            (e.g. message-id, Slack message ts, Notion page UUID).
            Used to deduplicate on re-sync.
        external_url: Human-readable link back to the original item,
            or empty string if not available.
        content_bytes: Raw binary content (e.g. PDF attachment bytes).
            Mutually exclusive with ``content_text``; set to ``None``
            when ``content_text`` is used.
        content_text: Decoded text content.  Mutually exclusive with
            ``content_bytes``; set to ``None`` when ``content_bytes``
            is used.
        mime_type: MIME type hint, e.g. ``"text/plain"``, ``"text/html"``,
            ``"application/pdf"``.  Defaults to ``"text/plain"``.
        metadata: Arbitrary provider-specific metadata dict.  Common keys:
            ``"title"``, ``"author"``, ``"created_at"``, ``"channel"``,
            ``"thread_id"``, ``"workspace"``.
        cursor_after: The provider cursor *after* this item.  The sync
            pipeline stores the cursor of the last successfully ingested
            artifact so it can resume from the right place on failure.
            Empty string if the provider does not emit per-item cursors.
    """

    source_type: str
    source_id: str
    external_url: str = ""
    content_bytes: Optional[bytes] = None
    content_text: Optional[str] = None
    mime_type: str = "text/plain"
    metadata: dict = field(default_factory=dict)
    cursor_after: str = ""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def has_content(self) -> bool:
        """Return True if at least one of ``content_text`` / ``content_bytes`` is set."""
        return self.content_text is not None or self.content_bytes is not None

    def as_text(self) -> str:
        """Return text content, decoding bytes as UTF-8 if necessary.

        Returns:
            Decoded string.  Empty string if neither field is set.
        """
        if self.content_text is not None:
            return self.content_text
        if self.content_bytes is not None:
            return self.content_bytes.decode("utf-8", errors="replace")
        return ""

    def as_dict(self) -> dict:
        """Return a serialisable dict (bytes encoded as length marker)."""
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "external_url": self.external_url,
            "content_bytes_len": len(self.content_bytes) if self.content_bytes else None,
            "content_text_len": len(self.content_text) if self.content_text else None,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
            "cursor_after": self.cursor_after,
        }
