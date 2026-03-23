"""PDF text parser using PyMuPDF (fitz) — extracts text blocks with page numbers."""

import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.parser import BaseParser

logger = logging.getLogger(__name__)

_PARSER_NAME = "pymupdf"
_SUPPORTED_MIME_TYPES = {"application/pdf"}

# Minimum character threshold to consider a text block non-empty
_MIN_BLOCK_CHARS = 3


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


class PdfPyMuPDFParser(BaseParser):
    """PDF parser that uses PyMuPDF to extract text blocks with page numbers.

    Each text block from PyMuPDF becomes an IRBlock with the correct 1-based
    page number. Blocks shorter than ``_MIN_BLOCK_CHARS`` characters are
    discarded to remove noise (e.g., stray punctuation).

    Usage::

        parser = PdfPyMuPDFParser()
        if parser.supports("application/pdf"):
            doc = parser.parse("/path/to/document.pdf")
    """

    def supports(self, mime_type: str) -> bool:
        """Return True if this parser handles the given MIME type.

        Args:
            mime_type: MIME type string to check.

        Returns:
            True for application/pdf.
        """
        return mime_type in _SUPPORTED_MIME_TYPES

    def parse(self, source_path: str) -> Document:
        """Parse a PDF file and extract text into a Document.

        Opens the PDF with PyMuPDF, iterates over pages, and extracts text
        blocks. Each block is assigned the 1-based page number of its origin
        page. Empty or near-empty blocks are discarded.

        Args:
            source_path: Absolute path to the PDF file.

        Returns:
            Document with IRBlocks (each carrying a page number) and a
            populated ParseReport.

        Raises:
            ValueError: If the file cannot be opened or read.
        """
        path = Path(source_path)
        try:
            pdf = fitz.open(source_path)
        except Exception as exc:
            raise ValueError(f"Cannot open PDF '{source_path}': {exc}") from exc

        blocks: list[IRBlock] = []

        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                page_number = page_index + 1  # 1-based

                # get_text("blocks") returns list of (x0,y0,x1,y1,text,block_no,block_type)
                # block_type 0 = text, 1 = image
                try:
                    raw_blocks = page.get_text("blocks")
                except Exception as exc:
                    logger.warning("Failed to extract text from page %d: %s", page_number, exc)
                    continue

                for raw in raw_blocks:
                    block_type_flag = raw[6] if len(raw) > 6 else 0
                    if block_type_flag != 0:
                        # Skip image blocks
                        continue

                    text = raw[4].strip() if len(raw) > 4 else ""
                    if len(text) < _MIN_BLOCK_CHARS:
                        continue

                    blocks.append(
                        IRBlock(
                            block_type=BlockType.PARAGRAPH,
                            text=text,
                            page=page_number,
                        )
                    )
        finally:
            pdf.close()

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
            mime_type="application/pdf",
            metadata={"filename": path.name, "extension": "pdf"},
            blocks=blocks,
            parse_report=report,
        )
