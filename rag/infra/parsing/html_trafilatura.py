"""HTML parser using Trafilatura — extracts main content and discards boilerplate."""

import hashlib
import logging
from pathlib import Path

import trafilatura

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.parser import BaseParser

logger = logging.getLogger(__name__)

_PARSER_NAME = "trafilatura"
_SUPPORTED_MIME_TYPES = {"text/html", "application/xhtml+xml"}


def _split_into_blocks(text: str) -> list[IRBlock]:
    """Split extracted plain text into paragraph IRBlocks.

    Blank lines are used as paragraph separators. Each non-empty paragraph
    becomes a single PARAGRAPH block.

    Args:
        text: Plain text output from trafilatura.

    Returns:
        List of IRBlock objects.
    """
    blocks: list[IRBlock] = []
    current: list[str] = []

    for line in text.splitlines():
        if line.strip():
            current.append(line)
        else:
            para = "\n".join(current).strip()
            if para:
                blocks.append(IRBlock(block_type=BlockType.PARAGRAPH, text=para))
            current = []

    para = "\n".join(current).strip()
    if para:
        blocks.append(IRBlock(block_type=BlockType.PARAGRAPH, text=para))

    return blocks


def _compute_non_printable_ratio(text: str) -> float:
    """Return fraction of non-printable characters in text.

    Args:
        text: Input string to analyse.

    Returns:
        Float in [0.0, 1.0].
    """
    if not text:
        return 0.0
    non_printable = sum(
        1 for ch in text if not ch.isprintable() and ch not in ("\n", "\r", "\t")
    )
    return non_printable / len(text)


def _compute_repetition_score(blocks: list[IRBlock]) -> float:
    """Return fraction of blocks that are exact duplicates of a prior block.

    Args:
        blocks: Ordered list of IRBlocks.

    Returns:
        Float in [0.0, 1.0].
    """
    if len(blocks) <= 1:
        return 0.0
    seen: set[str] = set()
    duplicates = 0
    for block in blocks:
        key = block.text.strip()
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates / len(blocks)


class HtmlTrafilaturaParser(BaseParser):
    """HTML parser that uses Trafilatura to extract main page content.

    Strips navigation, ads, headers, footers, and other boilerplate from
    HTML documents. The extracted plain text is then split into paragraph
    IRBlocks.

    Usage::

        parser = HtmlTrafilaturaParser()
        if parser.supports("text/html"):
            doc = parser.parse("/path/to/page.html")
    """

    def supports(self, mime_type: str) -> bool:
        """Return True if this parser handles the given MIME type.

        Args:
            mime_type: MIME type string to check.

        Returns:
            True for text/html and application/xhtml+xml.
        """
        return mime_type in _SUPPORTED_MIME_TYPES

    def parse(self, source_path: str) -> Document:
        """Parse an HTML file and extract main content into a Document.

        Args:
            source_path: Absolute path to the HTML file.

        Returns:
            Document with IRBlocks and a populated ParseReport.

        Raises:
            ValueError: If the file cannot be read or trafilatura extracts nothing.
        """
        path = Path(source_path)
        try:
            html_content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ValueError(f"Cannot read file '{source_path}': {exc}") from exc

        try:
            extracted = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )
        except Exception as exc:
            raise ValueError(f"Trafilatura failed on '{source_path}': {exc}") from exc

        if not extracted:
            logger.warning("Trafilatura extracted no content from '%s'", source_path)
            extracted = ""

        blocks = _split_into_blocks(extracted)

        all_text = " ".join(b.text for b in blocks)
        char_count = len(all_text)
        non_printable_ratio = _compute_non_printable_ratio(all_text)
        repetition_score = _compute_repetition_score(blocks)

        report = ParseReport(
            char_count=char_count,
            block_count=len(blocks),
            non_printable_ratio=non_printable_ratio,
            repetition_score=repetition_score,
            parser_used=_PARSER_NAME,
            fallback_triggered=False,
        )

        doc_id = hashlib.sha256(source_path.encode()).hexdigest()[:16]

        return Document(
            doc_id=doc_id,
            source_path=source_path,
            mime_type="text/html",
            metadata={"filename": path.name, "extension": path.suffix.lstrip(".")},
            blocks=blocks,
            parse_report=report,
        )
