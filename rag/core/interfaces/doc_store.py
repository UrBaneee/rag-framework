"""Abstract base class for document store plugins."""

from abc import ABC, abstractmethod
from typing import Optional

from rag.core.contracts.chunk import Chunk
from rag.core.contracts.document import Document
from rag.core.contracts.text_block import TextBlock


class BaseDocStore(ABC):
    """Interface that all document store plugins must implement.

    The DocStore persists documents, text blocks, and chunks. It is the
    primary storage layer for the ingestion pipeline. The V1 implementation
    uses SQLite; future implementations may use Postgres or other backends.

    Core tables (from Section 14):
        documents, text_blocks, chunks
    """

    @abstractmethod
    def save_document(self, document: Document) -> str:
        """Persist a Document and return its doc_id.

        Args:
            document: The parsed Document to store.

        Returns:
            The ``doc_id`` assigned to the stored document.
        """

    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Retrieve a Document by its doc_id.

        Args:
            doc_id: Unique document identifier.

        Returns:
            The Document if found, or None.
        """

    @abstractmethod
    def document_exists(self, doc_id: str) -> bool:
        """Check whether a document with the given doc_id exists.

        Args:
            doc_id: Unique document identifier.

        Returns:
            True if the document exists in the store.
        """

    @abstractmethod
    def save_text_blocks(self, blocks: list[TextBlock]) -> None:
        """Persist a list of TextBlocks.

        Args:
            blocks: TextBlocks to store. Each block's ``doc_id`` must
                reference an already-stored Document.
        """

    @abstractmethod
    def get_text_blocks(self, doc_id: str) -> list[TextBlock]:
        """Retrieve all TextBlocks for a document.

        Args:
            doc_id: Parent document identifier.

        Returns:
            Ordered list of TextBlocks for the document.
        """

    @abstractmethod
    def save_chunks(self, chunks: list[Chunk]) -> None:
        """Persist a list of Chunks.

        Args:
            chunks: Chunks to store. Each chunk's ``doc_id`` must
                reference an already-stored Document.
        """

    @abstractmethod
    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Retrieve all Chunks for a document.

        Args:
            doc_id: Parent document identifier.

        Returns:
            List of Chunks for the document.
        """

    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieve a single Chunk by its chunk_id.

        Args:
            chunk_id: Unique chunk identifier.

        Returns:
            The Chunk if found, or None.
        """

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its associated blocks and chunks.

        Args:
            doc_id: Unique document identifier to delete.
        """
