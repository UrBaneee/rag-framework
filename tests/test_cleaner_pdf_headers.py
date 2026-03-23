"""Tests for the PdfHeaderFooterDedupe cleaner."""

import pytest

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.infra.cleaning.pdf_header_footer_dedupe import PdfHeaderFooterDedupe


def _block(text: str, page: int | None = None) -> IRBlock:
    return IRBlock(block_type=BlockType.PARAGRAPH, text=text, page=page)


@pytest.fixture()
def cleaner() -> PdfHeaderFooterDedupe:
    return PdfHeaderFooterDedupe()


# ---------------------------------------------------------------------------
# Basic header/footer removal
# ---------------------------------------------------------------------------


def test_clean_removes_repeating_header(cleaner):
    # "Company Report" appears on pages 1, 2, 3 — all 3 pages → 100%
    blocks = [
        _block("Company Report", page=1),
        _block("Intro paragraph.", page=1),
        _block("Company Report", page=2),
        _block("Main content here.", page=2),
        _block("Company Report", page=3),
        _block("Conclusion text.", page=3),
    ]
    result = cleaner.clean(blocks)
    texts = [b.text for b in result]
    assert "Company Report" not in texts
    assert "Intro paragraph." in texts
    assert "Main content here." in texts


def test_clean_keeps_unique_content(cleaner):
    # Header on all 3 pages (100%) → removed
    # Unique content on 1/3 pages (33%) → kept (below 50% threshold)
    blocks = [
        _block("Header", page=1),
        _block("Unique content on page 1.", page=1),
        _block("Header", page=2),
        _block("Unique content on page 2.", page=2),
        _block("Header", page=3),
        _block("Unique content on page 3.", page=3),
    ]
    result = cleaner.clean(blocks)
    texts = [b.text for b in result]
    assert "Unique content on page 1." in texts
    assert "Unique content on page 2." in texts
    assert "Unique content on page 3." in texts


def test_clean_below_threshold_kept():
    # threshold=0.5: appears on 1/3 pages → below threshold → kept
    cleaner = PdfHeaderFooterDedupe(page_fraction_threshold=0.5)
    blocks = [
        _block("Rare header", page=1),
        _block("Content A", page=1),
        _block("Content B", page=2),
        _block("Content C", page=3),
    ]
    result = cleaner.clean(blocks)
    texts = [b.text for b in result]
    assert "Rare header" in texts


def test_clean_custom_threshold():
    # threshold=0.3: appears on 1/3 pages (0.33) → above threshold → removed
    cleaner = PdfHeaderFooterDedupe(page_fraction_threshold=0.3)
    blocks = [
        _block("Running title", page=1),
        _block("Content A", page=2),
        _block("Content B", page=3),
    ]
    result = cleaner.clean(blocks)
    texts = [b.text for b in result]
    assert "Running title" not in texts


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_clean_no_paged_blocks_passes_through(cleaner):
    blocks = [_block("No page", page=None), _block("Also no page", page=None)]
    result = cleaner.clean(blocks)
    assert result == blocks


def test_clean_single_page_passes_through(cleaner):
    # Only 1 page — heuristic requires >= 2 pages
    blocks = [
        _block("Header", page=1),
        _block("Content", page=1),
    ]
    result = cleaner.clean(blocks)
    assert len(result) == 2


def test_clean_empty_input(cleaner):
    assert cleaner.clean([]) == []


def test_clean_no_repeating_headers(cleaner):
    blocks = [
        _block("Unique A", page=1),
        _block("Unique B", page=2),
        _block("Unique C", page=3),
    ]
    result = cleaner.clean(blocks)
    assert len(result) == 3


def test_clean_preserves_non_repeating_metadata(cleaner):
    blocks = [
        _block("Footer", page=1),
        _block("Footer", page=2),
        IRBlock(block_type=BlockType.HEADING, text="Real Title", page=1),
        _block("Footer", page=3),
    ]
    result = cleaner.clean(blocks)
    headings = [b for b in result if b.block_type == BlockType.HEADING]
    assert len(headings) == 1
    assert headings[0].text == "Real Title"
