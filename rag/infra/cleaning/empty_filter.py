"""Empty-block filter cleaner — removes whitespace-only IRBlocks from the pipeline."""

from rag.core.contracts.ir_block import IRBlock
from rag.core.interfaces.cleaner import BaseCleaner


class EmptyBlockFilter(BaseCleaner):
    """Cleaner that removes empty or whitespace-only IRBlocks.

    A block is considered empty if its text, after stripping all whitespace,
    has fewer than ``min_chars`` characters (default 1). This removes blank
    paragraphs, stray newlines, and other artefacts left by parsers.

    Usage::

        cleaner = EmptyBlockFilter()
        cleaned_blocks = cleaner.clean(blocks)

        # Stricter threshold — remove blocks shorter than 10 chars
        cleaner = EmptyBlockFilter(min_chars=10)

    Args:
        min_chars: Minimum number of non-whitespace characters a block must
            contain to be kept. Defaults to 1.
    """

    def __init__(self, min_chars: int = 1) -> None:
        self._min_chars = min_chars

    def clean(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Remove empty or near-empty blocks from the list.

        Args:
            blocks: Input IRBlocks from the previous pipeline stage.

        Returns:
            Filtered list containing only blocks whose stripped text
            length is >= min_chars.
        """
        return [b for b in blocks if len(b.text.strip()) >= self._min_chars]
