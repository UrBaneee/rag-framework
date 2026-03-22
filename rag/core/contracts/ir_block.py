"""Intermediate representation block — the atomic unit output by all parsers."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BlockType(str, Enum):
    """Supported block content types."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE = "table"
    CODE = "code"
    IMAGE_CAPTION = "image_caption"
    FOOTER = "footer"
    HEADER = "header"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Page-coordinate bounding box (normalised 0–1 or raw points)."""

    x0: float
    y0: float
    x1: float
    y1: float


class IRBlock(BaseModel):
    """A single content block produced by a document parser.

    All parsers convert their native format into a list of IRBlocks so that
    downstream cleaning, chunking, and embedding stages are format-agnostic.

    Attributes:
        block_type: Semantic type of the content.
        text: Extracted plain text for this block.
        page: 1-based page number (None for formats without pages).
        bbox: Optional bounding box within the page.
        confidence: Parser confidence score in [0.0, 1.0]. 1.0 means certain.
        section_path: Ordered list of ancestor heading texts, e.g.
            ["Chapter 1", "Introduction"].
    """

    block_type: BlockType = BlockType.PARAGRAPH
    text: str
    page: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    section_path: list[str] = Field(default_factory=list)
