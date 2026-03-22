"""Parse quality report attached to every parsed Document."""

from pydantic import BaseModel, Field


class ParseReport(BaseModel):
    """Quality metrics produced by a parser after processing a document.

    Used by quality gates to decide whether to accept a parse result or
    trigger a fallback parser.

    Attributes:
        char_count: Total character count across all extracted blocks.
        block_count: Number of IRBlocks produced.
        non_printable_ratio: Fraction of characters that are non-printable
            (control characters, null bytes, etc.). High values indicate
            a poor parse.
        repetition_score: Fraction of blocks that are near-duplicates of
            another block in the same document. High values suggest
            repeated headers/footers were not cleaned.
        parser_used: Identifier of the parser that produced this result,
            e.g. "pymupdf", "trafilatura", "md_parser".
        fallback_triggered: True if an automatic fallback parser was
            invoked during this parse.
    """

    char_count: int = Field(ge=0)
    block_count: int = Field(ge=0)
    non_printable_ratio: float = Field(ge=0.0, le=1.0)
    repetition_score: float = Field(ge=0.0, le=1.0)
    parser_used: str
    fallback_triggered: bool = False
