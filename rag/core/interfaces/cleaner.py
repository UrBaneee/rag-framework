"""Abstract base class for cleaner pipeline steps."""

from abc import ABC, abstractmethod

from rag.core.contracts.ir_block import IRBlock


class BaseCleaner(ABC):
    """Interface for a single cleaning step in the cleaning pipeline.

    Cleaners operate on a list of ``IRBlock`` objects and return a filtered
    or transformed list. They are composed sequentially — the output of one
    cleaner is passed as input to the next.

    Implementations include:
    - unicode_fix: normalise Unicode characters
    - empty_filter: remove empty or whitespace-only blocks
    - dedupe: remove near-duplicate blocks
    - pdf_header_footer: strip repeated PDF headers and footers
    - html_nav_footer: strip HTML navigation and footer blocks
    - ocr_line_merge: reconstruct OCR-fragmented lines
    """

    @abstractmethod
    def clean(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Apply this cleaning step to a list of IRBlocks.

        Args:
            blocks: Input blocks from the previous pipeline stage.

        Returns:
            Cleaned blocks. May be shorter than input if blocks were
            removed, or contain modified text if normalisation was applied.
        """
