"""Tests for the PyMuPDF PDF text parser."""

from pathlib import Path

import fitz
import pytest

from rag.core.contracts.ir_block import BlockType
from rag.infra.parsing.pdf_pymupdf import PdfPyMuPDFParser

SAMPLE_PDF = Path(__file__).parent / "fixtures" / "sample.pdf"


@pytest.fixture()
def parser() -> PdfPyMuPDFParser:
    return PdfPyMuPDFParser()


def _make_pdf(tmp_path: Path, pages: list[list[str]]) -> str:
    """Create a PDF with specified text lines per page and return its path."""
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 100
        for line in lines:
            page.insert_text((72, y), line)
            y += 20
    path = tmp_path / "test.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


# ---------------------------------------------------------------------------
# supports()
# ---------------------------------------------------------------------------


def test_supports_application_pdf(parser):
    assert parser.supports("application/pdf") is True


def test_supports_html_returns_false(parser):
    assert parser.supports("text/html") is False


def test_supports_plain_text_returns_false(parser):
    assert parser.supports("text/plain") is False


# ---------------------------------------------------------------------------
# parse() — sample PDF
# ---------------------------------------------------------------------------


def test_parse_sample_pdf_returns_document(parser):
    doc = parser.parse(str(SAMPLE_PDF))
    assert doc.source_path == str(SAMPLE_PDF)
    assert doc.mime_type == "application/pdf"
    assert doc.doc_id


def test_parse_sample_pdf_has_blocks(parser):
    doc = parser.parse(str(SAMPLE_PDF))
    assert len(doc.blocks) > 0


def test_parse_sample_pdf_blocks_have_page_numbers(parser):
    doc = parser.parse(str(SAMPLE_PDF))
    for block in doc.blocks:
        assert block.page is not None
        assert block.page >= 1


def test_parse_sample_pdf_page_numbers_are_correct(parser):
    doc = parser.parse(str(SAMPLE_PDF))
    page_numbers = {b.page for b in doc.blocks}
    # Sample PDF has 2 pages
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_parse_sample_pdf_non_empty_text(parser):
    doc = parser.parse(str(SAMPLE_PDF))
    for block in doc.blocks:
        assert len(block.text.strip()) >= 3


# ---------------------------------------------------------------------------
# parse() — programmatically created PDFs
# ---------------------------------------------------------------------------


def test_parse_single_page_pdf(parser, tmp_path):
    path = _make_pdf(tmp_path, [["Hello world", "Second line of text here."]])
    doc = parser.parse(path)
    assert len(doc.blocks) > 0
    all_text = " ".join(b.text for b in doc.blocks)
    assert "Hello" in all_text or "world" in all_text or "Second" in all_text


def test_parse_multi_page_pdf_block_types(parser, tmp_path):
    path = _make_pdf(tmp_path, [
        ["Page one content line one.", "Page one content line two."],
        ["Page two content line one."],
    ])
    doc = parser.parse(path)
    for block in doc.blocks:
        assert block.block_type == BlockType.PARAGRAPH


def test_parse_multi_page_pdf_page_assignment(parser, tmp_path):
    path = _make_pdf(tmp_path, [
        ["First page paragraph."],
        ["Second page paragraph."],
    ])
    doc = parser.parse(path)
    pages = {b.page for b in doc.blocks}
    assert 1 in pages
    assert 2 in pages


# ---------------------------------------------------------------------------
# parse() — ParseReport
# ---------------------------------------------------------------------------


def test_parse_report_populated(parser):
    doc = parser.parse(str(SAMPLE_PDF))
    report = doc.parse_report
    assert report is not None
    assert report.parser_used == "pymupdf"
    assert report.block_count == len(doc.blocks)
    assert report.char_count > 0
    assert 0.0 <= report.non_printable_ratio <= 1.0
    assert 0.0 <= report.repetition_score <= 1.0
    assert report.fallback_triggered is False


# ---------------------------------------------------------------------------
# parse() — error handling
# ---------------------------------------------------------------------------


def test_parse_nonexistent_file_raises(parser):
    with pytest.raises(ValueError, match="Cannot open PDF"):
        parser.parse("/nonexistent/path/doc.pdf")


def test_parse_metadata_contains_filename(parser):
    doc = parser.parse(str(SAMPLE_PDF))
    assert doc.metadata["filename"] == "sample.pdf"
    assert doc.metadata["extension"] == "pdf"
