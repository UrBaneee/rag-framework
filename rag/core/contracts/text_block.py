"""TextBlock — a cleaned IRBlock stored in the DocStore."""

from typing import Optional

from pydantic import BaseModel, Field

from rag.core.contracts.ir_block import BlockType


class TextBlock(BaseModel):
    """A cleaned, deduplicated content block stored in the DocStore.

    TextBlocks are produced after the cleaning pipeline runs on IRBlocks.
    Each TextBlock is assigned a deterministic hash used for incremental
    ingestion (to detect unchanged blocks across document versions).

    Attributes:
        block_id: Surrogate primary key (assigned by the DocStore).
        doc_id: Parent document identifier.
        block_type: Semantic type inherited from the IRBlock.
        text: Cleaned plain text for this block.
        block_hash: SHA-256 of the canonicalised text (see hashing.py).
        page: 1-based page number, if applicable.
        sequence: 0-based position of this block within the document.
        section_path: Ordered ancestor heading texts.
    """

    block_id: Optional[str] = None
    doc_id: str
    block_type: BlockType = BlockType.PARAGRAPH
    text: str
    block_hash: str
    page: Optional[int] = None
    sequence: int = Field(ge=0)
    section_path: list[str] = Field(default_factory=list)
