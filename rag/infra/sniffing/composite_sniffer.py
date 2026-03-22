"""Composite sniffer — combines magic-byte and MIME detection strategies."""

from dataclasses import dataclass

from rag.infra.loading.local_file_loader import RawArtifact
from rag.infra.sniffing.sniffer_magic import MagicSniffer
from rag.infra.sniffing.sniffer_mime import MimeSniffer


# ── Zip container specialisation ──────────────────────────────────────────────
# PK magic matches DOCX, PPTX, XLSX, ODT, etc. Use extension to specialise.

_ZIP_EXTENSION_MAP: dict[str, tuple[str, str]] = {
    ".docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "unsupported",
    ),
    ".pptx": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "unsupported",
    ),
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "unsupported",
    ),
}


@dataclass
class SniffResult:
    """Final sniff result returned by the CompositeSniffer.

    Attributes:
        mime_type: Detected MIME type string (e.g. "application/pdf").
        detected_type: Stable short type identifier used for parser routing:
            "pdf", "html", "txt", "markdown", "unsupported", or "unknown".
        confidence: Detection confidence in [0.0, 1.0].
        strategy: Which strategy produced the result ("magic", "mime",
            "zip_specialised", or "unknown").
    """

    mime_type: str
    detected_type: str
    confidence: float
    strategy: str


class CompositeSniffer:
    """Combines magic-byte and extension-based detection strategies.

    Detection priority:
    1. Magic bytes (highest confidence)
    2. Zip-container specialisation (refine generic PK magic using extension)
    3. Extension / MIME fallback
    4. "unknown" (when nothing matches)

    Usage::

        sniffer = CompositeSniffer()
        result = sniffer.sniff(artifact)
        print(result.detected_type)  # "pdf", "html", "txt", "markdown", etc.
    """

    def __init__(self) -> None:
        self._magic = MagicSniffer()
        self._mime = MimeSniffer()

    def sniff(self, artifact: RawArtifact) -> SniffResult:
        """Detect the file type of a RawArtifact.

        Args:
            artifact: RawArtifact produced by the loader stage.

        Returns:
            A SniffResult with mime_type, detected_type, confidence,
            and strategy. detected_type is never None — falls back to
            "unknown" if no strategy succeeds.
        """
        # 1. Try magic bytes
        magic = self._magic.sniff(artifact)
        if magic.detected_type is not None:
            # 2. Specialise generic zip containers by extension
            if magic.detected_type == "zip_container":
                ext = artifact.metadata.get("extension", "").lower()
                if ext in _ZIP_EXTENSION_MAP:
                    mime_type, detected_type = _ZIP_EXTENSION_MAP[ext]
                    return SniffResult(
                        mime_type=mime_type,
                        detected_type=detected_type,
                        confidence=0.95,
                        strategy="zip_specialised",
                    )
                # Unknown zip — mark unsupported
                return SniffResult(
                    mime_type="application/zip",
                    detected_type="unsupported",
                    confidence=0.8,
                    strategy="magic",
                )
            return SniffResult(
                mime_type=magic.mime_type,
                detected_type=magic.detected_type,
                confidence=magic.confidence,
                strategy="magic",
            )

        # 3. Fall back to extension / MIME detection
        mime = self._mime.sniff(artifact)
        if mime.detected_type is not None:
            return SniffResult(
                mime_type=mime.mime_type,
                detected_type=mime.detected_type,
                confidence=mime.confidence,
                strategy="mime",
            )

        # 4. Unknown
        return SniffResult(
            mime_type="application/octet-stream",
            detected_type="unknown",
            confidence=0.0,
            strategy="unknown",
        )
