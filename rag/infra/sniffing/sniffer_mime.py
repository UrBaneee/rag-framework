"""MIME sniffer — detects file type from file extension."""

from dataclasses import dataclass
from typing import Optional

from rag.infra.loading.local_file_loader import RawArtifact


# ── Extension → (mime_type, detected_type) map ────────────────────────────────

_EXT_MAP: dict[str, tuple[str, str]] = {
    # Supported V1 types
    ".pdf":      ("application/pdf",  "pdf"),
    ".html":     ("text/html",        "html"),
    ".htm":      ("text/html",        "html"),
    ".txt":      ("text/plain",       "txt"),
    ".md":       ("text/markdown",    "markdown"),
    ".markdown": ("text/markdown",    "markdown"),
    # V2 types — parsers not yet implemented
    ".docx":     ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "unsupported"),
    ".pptx":     ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "unsupported"),
    ".xlsx":     ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "unsupported"),
    # Images
    ".png":      ("image/png",        "unsupported"),
    ".jpg":      ("image/jpeg",       "unsupported"),
    ".jpeg":     ("image/jpeg",       "unsupported"),
    ".gif":      ("image/gif",        "unsupported"),
}


@dataclass
class MimeSniffResult:
    """Result of an extension-based sniff attempt.

    Attributes:
        mime_type: Detected MIME type, or None if extension is unknown.
        detected_type: Stable short type string, or None if unknown.
        confidence: Extension-based detection is less reliable (0.7).
    """

    mime_type: Optional[str]
    detected_type: Optional[str]
    confidence: float = 0.7


class MimeSniffer:
    """Detects file type from the file extension in the artifact source path.

    Extension-based detection is a fallback when magic bytes are inconclusive.
    It is less reliable than magic-byte detection because extensions can be
    changed or absent.
    """

    def sniff(self, artifact: RawArtifact) -> MimeSniffResult:
        """Detect MIME type from the file extension.

        Args:
            artifact: RawArtifact with source_path populated.

        Returns:
            A MimeSniffResult with detected mime_type and detected_type,
            or None values if the extension is unknown.
        """
        ext = artifact.metadata.get("extension", "").lower()
        if ext in _EXT_MAP:
            mime_type, detected_type = _EXT_MAP[ext]
            return MimeSniffResult(
                mime_type=mime_type,
                detected_type=detected_type,
                confidence=0.7,
            )
        return MimeSniffResult(mime_type=None, detected_type=None, confidence=0.0)
