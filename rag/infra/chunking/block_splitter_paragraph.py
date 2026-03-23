"""Paragraph block splitter — converts cleaned IRBlocks into hashed TextBlocks."""

import hashlib
import unicodedata

from rag.core.contracts.ir_block import IRBlock
from rag.core.contracts.text_block import TextBlock
from rag.core.interfaces.block_splitter import BaseBlockSplitter


def _compute_block_hash(text: str) -> str:
    """Compute a stable SHA-256 hash for a block's canonical text.

    Canonicalisation steps:
    1. Unicode NFC normalisation.
    2. Strip leading/trailing whitespace.
    3. Collapse internal whitespace runs to a single space.

    This ensures the hash is stable across minor formatting differences
    (e.g., double spaces, leading/trailing newlines).

    Args:
        text: Raw block text.

    Returns:
        Hex-encoded SHA-256 digest (64 characters).
    """
    normalised = unicodedata.normalize("NFC", text).strip()
    canonical = " ".join(normalised.split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ParagraphBlockSplitter(BaseBlockSplitter):
    """Block splitter that maps each cleaned IRBlock to a single TextBlock.

    Each IRBlock becomes exactly one TextBlock. The splitter assigns:
    - a deterministic ``block_hash`` computed from canonicalised text
    - a 0-based ``sequence`` number reflecting the block's position
    - the ``doc_id``, ``block_type``, ``page``, and ``section_path``
      inherited from the source IRBlock

    Blocks with empty text (after stripping) are skipped.

    Usage::

        splitter = ParagraphBlockSplitter()
        text_blocks = splitter.split(doc_id="abc123", blocks=ir_blocks)
    """

    def split(self, doc_id: str, blocks: list[IRBlock]) -> list[TextBlock]:
        """Convert cleaned IRBlocks into sequenced, hashed TextBlocks.

        Args:
            doc_id: Parent document identifier embedded in each TextBlock.
            blocks: Cleaned IRBlocks from the cleaning pipeline.

        Returns:
            Ordered list of TextBlocks. Blocks with empty text are omitted.
            Sequence numbers are contiguous starting from 0.
        """
        result: list[TextBlock] = []
        sequence = 0

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            result.append(
                TextBlock(
                    doc_id=doc_id,
                    block_type=block.block_type,
                    text=text,
                    block_hash=_compute_block_hash(text),
                    page=block.page,
                    sequence=sequence,
                    section_path=list(block.section_path),
                )
            )
            sequence += 1

        return result
