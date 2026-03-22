"""Top-level Document contract — the output of the ingestion parsing stage."""

from typing import Any, Optional

from pydantic import BaseModel, Field

from rag.core.contracts.ir_block import IRBlock
from rag.core.contracts.parse_report import ParseReport


class Document(BaseModel):
    """A parsed document ready for cleaning and chunking.

    Produced by parser plugins and passed through the cleaning pipeline
    before being split into Chunks for indexing.

    Attributes:
        doc_id: Stable unique identifier for this document (e.g. SHA-256 of
            source path + content hash).
        source_path: Absolute or relative path to the source file.
        mime_type: MIME type detected by the sniffer, e.g. "application/pdf".
        metadata: Arbitrary key-value pairs extracted from the document
            (title, author, creation date, page count, etc.).
        blocks: Ordered list of IRBlocks produced by the parser.
        parse_report: Quality metrics for this parse result.
    """

    doc_id: str
    source_path: str
    mime_type: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    blocks: list[IRBlock] = Field(default_factory=list)
    parse_report: Optional[ParseReport] = None
