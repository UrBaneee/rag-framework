"""Abstract base class for chunk packer (TextBlock → Chunk conversion)."""

from abc import ABC, abstractmethod

from rag.core.contracts.chunk import Chunk
from rag.core.contracts.text_block import TextBlock


class BaseChunkPacker(ABC):
    """Interface for packing TextBlocks into indexable Chunks.

    The chunk packer groups consecutive TextBlocks into Chunks that fit
    within a token budget. It also assigns:
    - ``stable_text``: canonicalised text used for embedding / BM25
    - ``display_text``: human-readable text for UI and citations
    - ``chunk_signature``: SHA-256 over the ordered block hashes

    Different strategies (paragraph-boundary, token-window, semantic)
    can be swapped by providing different implementations.
    """

    @abstractmethod
    def pack(self, blocks: list[TextBlock]) -> list[Chunk]:
        """Pack TextBlocks into Chunks.

        Args:
            blocks: Ordered TextBlocks from the block splitter.

        Returns:
            List of Chunks ready for embedding and indexing.
        """
