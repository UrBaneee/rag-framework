"""Duplicate-paragraph cleaner — removes exact-duplicate IRBlocks."""

from rag.core.contracts.ir_block import IRBlock
from rag.core.interfaces.cleaner import BaseCleaner


class DedupeParagraphs(BaseCleaner):
    """Cleaner that removes exact-duplicate IRBlocks from a document.

    Two blocks are considered duplicates when their stripped text is
    identical. Only the first occurrence is kept; subsequent duplicates
    are discarded. Order of non-duplicate blocks is preserved.

    Usage::

        cleaner = DedupeParagraphs()
        cleaned_blocks = cleaner.clean(blocks)
    """

    def clean(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Remove duplicate blocks, keeping the first occurrence of each.

        Args:
            blocks: Input IRBlocks from the previous pipeline stage.

        Returns:
            Deduplicated list with original order preserved.
        """
        seen: set[str] = set()
        result: list[IRBlock] = []
        for block in blocks:
            key = block.text.strip()
            if key not in seen:
                seen.add(key)
                result.append(block)
        return result
