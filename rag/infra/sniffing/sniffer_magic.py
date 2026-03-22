"""Magic-byte sniffer — detects file type from raw content headers."""

from dataclasses import dataclass
from typing import Optional

from rag.infra.loading.local_file_loader import RawArtifact


# ── Known magic byte signatures ────────────────────────────────────────────────

# Each entry: (prefix_bytes, mime_type, detected_type)
_MAGIC_SIGNATURES: list[tuple[bytes, str, str]] = [
    # PDF: starts with %PDF
    (b"%PDF", "application/pdf", "pdf"),
    # PK zip header — used by DOCX, PPTX, XLSX, ODT, etc.
    (b"PK\x03\x04", "application/zip", "zip_container"),
    # PNG image
    (b"\x89PNG\r\n\x1a\n", "image/png", "unsupported"),
    # JPEG image
    (b"\xff\xd8\xff", "image/jpeg", "unsupported"),
    # GIF image
    (b"GIF87a", "image/gif", "unsupported"),
    (b"GIF89a", "image/gif", "unsupported"),
]

# HTML heuristics — check start of decoded text
_HTML_PREFIXES = (
    b"<!doctype html",
    b"<!DOCTYPE html",
    b"<!DOCTYPE HTML",
    b"<html",
    b"<HTML",
)


@dataclass
class MagicSniffResult:
    """Result of a magic-byte sniff attempt.

    Attributes:
        mime_type: Detected MIME type, or None if undetected.
        detected_type: Stable short type string ("pdf", "html", "unsupported",
            etc.), or None if undetected.
        confidence: Detection confidence (1.0 = certain, 0.5 = heuristic).
    """

    mime_type: Optional[str]
    detected_type: Optional[str]
    confidence: float = 1.0


class MagicSniffer:
    """Detects file type using magic bytes (file content headers).

    Magic-byte detection is format-specific and does not depend on
    file extensions, making it robust against mislabelled files.
    """

    def sniff(self, artifact: RawArtifact) -> MagicSniffResult:
        """Attempt to detect the file type from raw bytes.

        Args:
            artifact: RawArtifact with raw_bytes populated.

        Returns:
            A MagicSniffResult with detected mime_type and detected_type,
            or None values if the type could not be determined.
        """
        data = artifact.raw_bytes

        # Check known binary magic signatures
        for prefix, mime_type, detected_type in _MAGIC_SIGNATURES:
            if data.startswith(prefix):
                return MagicSniffResult(
                    mime_type=mime_type,
                    detected_type=detected_type,
                    confidence=1.0,
                )

        # HTML heuristic: check first 512 bytes (case-insensitive prefix match)
        head = data[:512].lstrip()
        for prefix in _HTML_PREFIXES:
            if head.startswith(prefix):
                return MagicSniffResult(
                    mime_type="text/html",
                    detected_type="html",
                    confidence=0.95,
                )

        # No match
        return MagicSniffResult(mime_type=None, detected_type=None, confidence=0.0)
