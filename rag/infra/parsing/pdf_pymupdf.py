"""PDF text parser using PyMuPDF (fitz) — extracts text blocks with page numbers.

Heading detection uses two complementary signals, operating at the **line level**
so that section headers embedded inside larger layout blocks are correctly split
out rather than merged with their body content.

Signal 1 — font size: a line whose dominant font size is at least
``_HEADING_FONT_RATIO`` × the document's character-weighted median font size is
classified as a heading.  This catches name headers, chapter titles, etc.

Signal 2 — ALL-CAPS pattern: a standalone line that is entirely upper-case
letters (plus spaces, ``&``, ``/``, ``-``) and between 4 and 60 characters long
is classified as a heading.  This catches section labels such as
"PROFESSIONAL EXPERIENCE", "EDUCATION", and "RELEVANT SKILLS" even when they
are rendered in the same font size as body text (only bold).

Processing is line-by-line: when a heading line is encountered inside a
PyMuPDF layout block, any accumulated paragraph lines are flushed as a
PARAGRAPH IRBlock first, then the heading line becomes its own HEADING IRBlock.
This ensures that section labels are never merged with the content that follows.
"""

import hashlib
import logging
import re
import statistics
from pathlib import Path

import fitz  # PyMuPDF

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.parser import BaseParser

logger = logging.getLogger(__name__)

_PARSER_NAME = "pymupdf"
_SUPPORTED_MIME_TYPES = {"application/pdf"}

# Minimum character threshold to consider a text block non-empty.
_MIN_BLOCK_CHARS = 3

# A line is a heading when its dominant font size >= median * this ratio.
_HEADING_FONT_RATIO = 1.2

# ALL-CAPS heading pattern: entire line is uppercase letters, spaces, and a
# small set of punctuation characters.  Min 4 chars to skip abbreviations like
# "NC" or "VA"; max 60 chars to skip sentences accidentally typed in caps.
_ALLCAPS_HEADING_RE = re.compile(r"^[A-Z][A-Z\s&/\-]{3,59}$")


# ── Font-size helpers ─────────────────────────────────────────────────────────

def _collect_all_spans(pdf: fitz.Document) -> list[tuple[float, int]]:
    """Collect (font_size, char_count) for every text span in the PDF.

    Used to compute the document-wide median body-text font size.

    Args:
        pdf: Open PyMuPDF Document.

    Returns:
        List of (font_size, char_count) tuples for non-empty spans.
    """
    result: list[tuple[float, int]] = []
    for page_index in range(len(pdf)):
        page = pdf[page_index]
        try:
            page_dict = page.get_text("dict")
        except Exception:
            continue
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0.0)
                    text = span.get("text", "")
                    n = len(text.strip())
                    if size > 0 and n > 0:
                        result.append((size, n))
    return result


def _median_font_size(spans: list[tuple[float, int]]) -> float:
    """Compute the character-weighted median font size across all spans.

    Weighting by character count prevents a few large-font title spans from
    skewing the median upward.

    Args:
        spans: List of (font_size, char_count) tuples.

    Returns:
        Median font size (12.0 if no spans are available).
    """
    if not spans:
        return 12.0
    expanded: list[float] = []
    for size, count in spans:
        expanded.extend([size] * count)
    return statistics.median(expanded) if expanded else 12.0


def _line_dominant_font_size(line: dict) -> float:
    """Return the font size used by the most characters in a single line.

    Args:
        line: PyMuPDF line dict from ``get_text("dict")``.

    Returns:
        Dominant font size, or 0.0 if no spans are found.
    """
    size_chars: dict[float, int] = {}
    for span in line.get("spans", []):
        size = span.get("size", 0.0)
        text = span.get("text", "")
        n = len(text.strip())
        if size > 0 and n > 0:
            size_chars[size] = size_chars.get(size, 0) + n
    if not size_chars:
        return 0.0
    return max(size_chars, key=lambda s: size_chars[s])


def _line_text(line: dict) -> str:
    """Concatenate all span texts in a line."""
    return "".join(span.get("text", "") for span in line.get("spans", []))


def _is_heading_line(text: str, dom_size: float, heading_threshold: float) -> bool:
    """Return True if a line should be classified as a heading.

    Two independent signals:
    1. Font-size: dominant size >= heading_threshold (document median * ratio).
    2. ALL-CAPS pattern: entire line matches ``_ALLCAPS_HEADING_RE``.

    Args:
        text: Stripped line text.
        dom_size: Dominant font size for this line (0.0 if unknown).
        heading_threshold: Font-size cutoff (median * _HEADING_FONT_RATIO).

    Returns:
        True if the line is a heading by either signal.
    """
    if not text:
        return False
    if dom_size > 0 and dom_size >= heading_threshold:
        return True
    if _ALLCAPS_HEADING_RE.match(text):
        return True
    return False


# ── Quality metrics ───────────────────────────────────────────────────────────

def _compute_non_printable_ratio(text: str) -> float:
    """Return fraction of non-printable characters in text."""
    if not text:
        return 0.0
    non_printable = sum(
        1 for ch in text if not ch.isprintable() and ch not in ("\n", "\r", "\t")
    )
    return non_printable / len(text)


def _compute_repetition_score(blocks: list[IRBlock]) -> float:
    """Return fraction of blocks that are exact duplicates of a prior block."""
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


# ── Parser ────────────────────────────────────────────────────────────────────

class PdfPyMuPDFParser(BaseParser):
    """PDF parser that uses PyMuPDF to extract text blocks with page numbers.

    Heading detection operates at the line level using two complementary
    signals: font size and ALL-CAPS text patterns.  When a heading line is
    found inside a larger layout block, accumulated paragraph lines are flushed
    first so the heading is never merged with its surrounding content.

    Usage::

        parser = PdfPyMuPDFParser()
        if parser.supports("application/pdf"):
            doc = parser.parse("/path/to/document.pdf")
    """

    def supports(self, mime_type: str) -> bool:
        """Return True if this parser handles the given MIME type."""
        return mime_type in _SUPPORTED_MIME_TYPES

    def parse(self, source_path: str) -> Document:
        """Parse a PDF file and extract text into a Document.

        Algorithm:
        1. Collect all text spans to compute the document-wide median font size
           and heading threshold.
        2. Iterate lines within each PyMuPDF layout block.  For each line:
           - If the line is a heading (font-size or ALL-CAPS signal), flush any
             accumulated paragraph lines as a PARAGRAPH IRBlock, then emit the
             heading line as a HEADING IRBlock.
           - Otherwise, accumulate the line into the current paragraph buffer.
        3. At the end of each layout block, flush any remaining paragraph lines.

        Args:
            source_path: Absolute path to the PDF file.

        Returns:
            Document with IRBlocks and a populated ParseReport.

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
            # ── Pass 1: compute document-wide median font size ─────────────
            all_spans = _collect_all_spans(pdf)
            median_size = _median_font_size(all_spans)
            heading_threshold = median_size * _HEADING_FONT_RATIO

            logger.debug(
                "PDF font analysis for '%s': median=%.1fpt, heading_threshold=%.1fpt",
                path.name,
                median_size,
                heading_threshold,
            )

            # ── Pass 2: line-level extraction with heading splitting ────────
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                page_number = page_index + 1

                try:
                    page_dict = page.get_text("dict")
                except Exception as exc:
                    logger.warning(
                        "Failed to extract text from page %d: %s", page_number, exc
                    )
                    continue

                for raw_block in page_dict.get("blocks", []):
                    if raw_block.get("type") != 0:
                        continue  # skip image blocks

                    pending_para_lines: list[str] = []

                    for line in raw_block.get("lines", []):
                        line_raw = _line_text(line)
                        line_stripped = line_raw.strip()
                        if not line_stripped:
                            continue

                        dom_size = _line_dominant_font_size(line)
                        is_heading = _is_heading_line(
                            line_stripped, dom_size, heading_threshold
                        )

                        if is_heading:
                            # Flush any accumulated paragraph content first
                            if pending_para_lines:
                                para_text = "\n".join(pending_para_lines).strip()
                                if len(para_text) >= _MIN_BLOCK_CHARS:
                                    blocks.append(
                                        IRBlock(
                                            block_type=BlockType.PARAGRAPH,
                                            text=para_text,
                                            page=page_number,
                                        )
                                    )
                                pending_para_lines = []

                            # Emit heading as its own IRBlock
                            if len(line_stripped) >= _MIN_BLOCK_CHARS:
                                logger.debug(
                                    "Heading detected (%.1fpt, caps=%s): %r",
                                    dom_size,
                                    bool(_ALLCAPS_HEADING_RE.match(line_stripped)),
                                    line_stripped[:60],
                                )
                                blocks.append(
                                    IRBlock(
                                        block_type=BlockType.HEADING,
                                        text=line_stripped,
                                        page=page_number,
                                    )
                                )
                        else:
                            pending_para_lines.append(line_raw)

                    # Flush remaining paragraph lines for this layout block
                    if pending_para_lines:
                        para_text = "\n".join(pending_para_lines).strip()
                        if len(para_text) >= _MIN_BLOCK_CHARS:
                            blocks.append(
                                IRBlock(
                                    block_type=BlockType.PARAGRAPH,
                                    text=para_text,
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
