"""Abstract base class for document parser plugins."""

from abc import ABC, abstractmethod

from rag.core.contracts.document import Document


class BaseParser(ABC):
    """Interface that all document parser plugins must implement.

    Parser plugins convert a raw source file into a ``Document`` containing
    an ordered list of ``IRBlock`` objects. Each parser handles one or more
    MIME types and must report which types it supports via ``supports()``.

    The pipeline uses ``supports()`` to route documents to the correct
    parser before calling ``parse()``.
    """

    @abstractmethod
    def supports(self, mime_type: str) -> bool:
        """Return True if this parser can handle the given MIME type.

        Args:
            mime_type: MIME type string, e.g. "application/pdf".

        Returns:
            True if this parser supports the MIME type.
        """

    @abstractmethod
    def parse(self, source_path: str) -> Document:
        """Parse a source file and return a structured Document.

        Args:
            source_path: Absolute path to the file to parse.

        Returns:
            A Document with populated blocks and parse_report.

        Raises:
            ValueError: If the file cannot be read or parsed.
        """
