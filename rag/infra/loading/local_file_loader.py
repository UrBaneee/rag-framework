"""Local file loader — reads a file from disk into a RawArtifact."""

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RawArtifact:
    """Raw file content and metadata produced by the loader stage.

    Passed from the loader to the sniffer, which detects the MIME type
    and routes it to the appropriate parser.

    Attributes:
        source_path: Absolute path to the source file.
        raw_bytes: Raw file content as bytes.
        metadata: Basic file metadata (size, name, extension, timestamps).
    """

    source_path: str
    raw_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Decode raw_bytes as UTF-8, replacing undecodable bytes.

        Returns:
            Decoded text string.
        """
        return self.raw_bytes.decode("utf-8", errors="replace")

    @property
    def size_bytes(self) -> int:
        """Size of the raw content in bytes."""
        return len(self.raw_bytes)


class LocalFileLoader:
    """Loads a local file from disk into a RawArtifact.

    Usage::

        loader = LocalFileLoader()
        artifact = loader.load("/path/to/document.pdf")
    """

    def load(self, path: str | Path) -> RawArtifact:
        """Read a file and return a RawArtifact with content and metadata.

        Args:
            path: Absolute or relative path to the file to load.

        Returns:
            A RawArtifact containing the raw bytes and file metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            IsADirectoryError: If the path points to a directory.
            PermissionError: If the file cannot be read.
            OSError: For other I/O errors.
        """
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")

        stat = file_path.stat()
        raw_bytes = file_path.read_bytes()

        metadata: dict[str, Any] = {
            "file_name": file_path.name,
            "extension": file_path.suffix.lower(),
            "file_size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
            "absolute_path": str(file_path),
        }

        return RawArtifact(
            source_path=str(file_path),
            raw_bytes=raw_bytes,
            metadata=metadata,
        )
