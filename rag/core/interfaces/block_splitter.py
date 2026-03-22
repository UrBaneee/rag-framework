"""Abstract base class for block splitter (IR → TextBlock conversion)."""

from abc import ABC, abstractmethod

from rag.core.contracts.ir_block import IRBlock
from rag.core.contracts.text_block import TextBlock


class BaseBlockSplitter(ABC):
    """Interface for converting cleaned IRBlocks into stored TextBlocks.

    The block splitter is responsible for:
    - splitting large IRBlocks into paragraph-sized TextBlocks
    - assigning a deterministic ``block_hash`` to each TextBlock
    - setting the ``sequence`` field for ordering

    The output of the block splitter is passed to the ``BaseChunkPacker``.
    """

    @abstractmethod
    def split(self, doc_id: str, blocks: list[IRBlock]) -> list[TextBlock]:
        """Convert cleaned IRBlocks into sequenced, hashed TextBlocks.

        Args:
            doc_id: Parent document identifier to embed in each TextBlock.
            blocks: Cleaned IRBlocks from the cleaning pipeline.

        Returns:
            Ordered list of TextBlocks ready for chunk packing.
        """
