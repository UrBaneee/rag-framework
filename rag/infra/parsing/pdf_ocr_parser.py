"""Scanned PDF parser using page rendering + OCR — Task 13.3.

Renders each PDF page to an image using ``BasePageRenderer``, then runs OCR
via ``BaseOCRProvider`` to extract text blocks.  Produces a ``Document``
with ``IRBlock`` objects that include page number, bounding box, and
per-block confidence scores.  The ``ParseReport`` records ``parser_used``
as ``"pdf_ocr"`` and includes aggregate per-page confidence statistics.

Usage::

    from rag.infra.ocr.renderer_pymupdf import PyMuPDFPageRenderer
    from rag.infra.ocr.paddleocr_provider import PaddleOCRProvider
    from rag.infra.parsing.pdf_ocr_parser import PdfOCRParser

    parser = PdfOCRParser(
        renderer=PyMuPDFPageRenderer(dpi=150),
        ocr_provider=PaddleOCRProvider(lang="en"),
    )
    document = parser.parse("/path/to/scanned.pdf")
"""

from __future__ import annotations

import logging
from pathlib import Path
from statistics import mean
from typing import Optional

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.ocr_provider import BaseOCRProvider
from rag.core.interfaces.page_renderer import BasePageRenderer
from rag.core.interfaces.parser import BaseParser

logger = logging.getLogger(__name__)


def _repetition_score(blocks: list[IRBlock]) -> float:
    """Fraction of blocks whose text is an exact duplicate of another."""
    if len(blocks) < 2:
        return 0.0
    texts = [b.text.strip() for b in blocks]
    seen: set[str] = set()
    dupes = 0
    for t in texts:
        if t in seen:
            dupes += 1
        seen.add(t)
    return dupes / len(texts)


class PdfOCRParser(BaseParser):
    """Parse scanned PDF files by rendering pages and running OCR.

    Args:
        renderer: A ``BasePageRenderer`` implementation (e.g. PyMuPDF).
        ocr_provider: A ``BaseOCRProvider`` implementation (e.g. PaddleOCR).
        max_pages: Maximum number of pages to process.  Defaults to None
            (process all pages).

    Attributes:
        PARSER_ID: Identifier written to ``ParseReport.parser_used``.
    """

    PARSER_ID = "pdf_ocr"

    def __init__(
        self,
        renderer: BasePageRenderer,
        ocr_provider: BaseOCRProvider,
        max_pages: Optional[int] = None,
    ) -> None:
        self._renderer = renderer
        self._ocr = ocr_provider
        self._max_pages = max_pages

    def supports(self, mime_type: str) -> bool:
        """Return True for PDF MIME types."""
        return mime_type in ("application/pdf", "application/x-pdf")

    def parse(self, source_path: str) -> Document:
        """Parse a scanned PDF by rendering pages to images and running OCR.

        Args:
            source_path: Absolute path to the scanned PDF.

        Returns:
            ``Document`` with:
            - ``IRBlock`` per OCR text region (with page, bbox, confidence)
            - ``ParseReport`` with ``parser_used="pdf_ocr"`` and aggregate
              confidence and per-page char counts in metadata
        """
        source_path = str(Path(source_path).resolve())
        n_pages = self._renderer.page_count(source_path)
        end_page = n_pages if self._max_pages is None else min(n_pages, self._max_pages)

        all_blocks: list[IRBlock] = []
        confidences: list[float] = []
        page_char_counts: list[int] = []

        for page_num in range(1, end_page + 1):
            image = self._renderer.render(source_path, page_num)
            page_blocks = self._ocr.ocr(image)

            page_chars = 0
            for block in page_blocks:
                # Stamp the page number (OCR provider doesn't know it)
                block = block.model_copy(update={"page": page_num})
                all_blocks.append(block)
                confidences.append(block.confidence)
                page_chars += len(block.text)

            page_char_counts.append(page_chars)
            logger.debug(
                "OCR page %d/%d: %d blocks, %d chars",
                page_num, end_page, len(page_blocks), page_chars,
            )

        mean_confidence = mean(confidences) if confidences else 0.0
        total_chars = sum(page_char_counts)

        # Non-printable ratio: hard to compute from OCR output — use 0.0
        non_printable_ratio = 0.0

        report = ParseReport(
            char_count=total_chars,
            block_count=len(all_blocks),
            non_printable_ratio=non_printable_ratio,
            repetition_score=_repetition_score(all_blocks),
            parser_used=self.PARSER_ID,
            fallback_triggered=False,
        )

        # Build a stub doc_id from the path (fingerprint assigned by pipeline)
        from hashlib import sha256
        doc_id = sha256(source_path.encode()).hexdigest()

        return Document(
            doc_id=doc_id,
            source_path=source_path,
            mime_type="application/pdf",
            blocks=all_blocks,
            parse_report=report,
            metadata={
                "mean_ocr_confidence": round(mean_confidence, 4),
                "page_count": end_page,
                "page_char_counts": page_char_counts,
            },
        )
