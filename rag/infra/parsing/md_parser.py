"""TXT/Markdown parser — converts plain text and Markdown into IRBlocks."""

import hashlib
import logging
from pathlib import Path

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.parser import BaseParser

logger = logging.getLogger(__name__)

_PARSER_NAME = "md_parser"
_SUPPORTED_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
}


def _compute_non_printable_ratio(text: str) -> float:
    """Return fraction of non-printable characters in text.

    Args:
        text: Input string to analyse.

    Returns:
        Float in [0.0, 1.0]; 0.0 means all printable.
    """
    if not text:
        return 0.0
    non_printable = sum(1 for ch in text if not ch.isprintable() and ch not in ("\n", "\r", "\t"))
    return non_printable / len(text)


def _compute_repetition_score(blocks: list[IRBlock]) -> float:
    """Return fraction of blocks that are exact duplicates of a prior block.

    Args:
        blocks: Ordered list of IRBlocks.

    Returns:
        Float in [0.0, 1.0]; 0.0 means no repetition.
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


def _parse_blocks(content: str) -> list[IRBlock]:
    """Convert raw text/markdown content into a list of IRBlocks.

    Parsing rules:
    - Lines starting with one or more ``#`` characters → HEADING.
    - Lines enclosed in triple-backtick fences → CODE (collected as one block).
    - Blank lines act as paragraph separators.
    - Everything else accumulates into PARAGRAPH blocks.

    Args:
        content: Raw file content as a string.

    Returns:
        Ordered list of IRBlocks (empty blocks are discarded).
    """
    blocks: list[IRBlock] = []
    lines = content.splitlines()

    in_code_fence = False
    code_lines: list[str] = []
    paragraph_lines: list[str] = []
    section_path: list[str] = []

    def _flush_paragraph() -> None:
        text = "\n".join(paragraph_lines).strip()
        if text:
            blocks.append(
                IRBlock(
                    block_type=BlockType.PARAGRAPH,
                    text=text,
                    section_path=list(section_path),
                )
            )
        paragraph_lines.clear()

    def _flush_code(fence_lines: list[str]) -> None:
        text = "\n".join(fence_lines).strip()
        if text:
            blocks.append(
                IRBlock(
                    block_type=BlockType.CODE,
                    text=text,
                    section_path=list(section_path),
                )
            )
        fence_lines.clear()

    for line in lines:
        # Code fence toggle
        if line.startswith("```"):
            if in_code_fence:
                _flush_code(code_lines)
                in_code_fence = False
            else:
                _flush_paragraph()
                in_code_fence = True
            continue

        if in_code_fence:
            code_lines.append(line)
            continue

        # Heading detection
        stripped = line.lstrip()
        if stripped.startswith("#"):
            _flush_paragraph()
            heading_text = stripped.lstrip("#").strip()
            if heading_text:
                level = len(stripped) - len(stripped.lstrip("#"))
                # Maintain section_path up to this heading level
                section_path = section_path[: level - 1] + [heading_text]
                blocks.append(
                    IRBlock(
                        block_type=BlockType.HEADING,
                        text=heading_text,
                        section_path=list(section_path[:-1]),
                    )
                )
            continue

        # Blank line → flush accumulated paragraph
        if not line.strip():
            _flush_paragraph()
            continue

        paragraph_lines.append(line)

    # Flush any remaining content
    if in_code_fence:
        _flush_code(code_lines)
    _flush_paragraph()

    return blocks


class MdParser(BaseParser):
    """Parser for plain text (.txt) and Markdown (.md) files.

    Converts the file into a ``Document`` containing ``IRBlock`` objects
    organised by headings, paragraphs, and fenced code blocks.

    Usage::

        parser = MdParser()
        if parser.supports("text/plain"):
            doc = parser.parse("/path/to/file.md")
    """

    def supports(self, mime_type: str) -> bool:
        """Return True if this parser handles the given MIME type.

        Args:
            mime_type: MIME type string to check.

        Returns:
            True for text/plain, text/markdown, text/x-markdown.
        """
        return mime_type in _SUPPORTED_MIME_TYPES

    def parse(self, source_path: str) -> Document:
        """Parse a TXT or Markdown file into a Document.

        Args:
            source_path: Absolute path to the source file.

        Returns:
            Document with IRBlocks and a populated ParseReport.

        Raises:
            ValueError: If the file cannot be read.
        """
        path = Path(source_path)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ValueError(f"Cannot read file '{source_path}': {exc}") from exc

        blocks = _parse_blocks(content)

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
        mime_type = "text/markdown" if path.suffix.lower() in (".md", ".markdown") else "text/plain"

        return Document(
            doc_id=doc_id,
            source_path=source_path,
            mime_type=mime_type,
            metadata={"filename": path.name, "extension": path.suffix.lstrip(".")},
            blocks=blocks,
            parse_report=report,
        )
