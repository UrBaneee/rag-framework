"""Unit tests for core document contracts: Document, IRBlock, ParseReport."""

import pytest

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, BoundingBox, IRBlock
from rag.core.contracts.parse_report import ParseReport


@pytest.mark.unit
class TestIRBlock:
    def test_irblock_minimal(self):
        block = IRBlock(text="Hello world")
        assert block.text == "Hello world"
        assert block.block_type == BlockType.PARAGRAPH
        assert block.confidence == 1.0
        assert block.section_path == []
        assert block.page is None
        assert block.bbox is None

    def test_irblock_with_all_fields(self):
        bbox = BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.3)
        block = IRBlock(
            block_type=BlockType.HEADING,
            text="Introduction",
            page=1,
            bbox=bbox,
            confidence=0.95,
            section_path=["Chapter 1"],
        )
        assert block.block_type == BlockType.HEADING
        assert block.page == 1
        assert block.bbox.x0 == 0.1
        assert block.confidence == 0.95
        assert block.section_path == ["Chapter 1"]

    def test_irblock_confidence_bounds(self):
        with pytest.raises(Exception):
            IRBlock(text="bad", confidence=1.5)
        with pytest.raises(Exception):
            IRBlock(text="bad", confidence=-0.1)

    def test_block_type_enum_values(self):
        assert BlockType.PARAGRAPH == "paragraph"
        assert BlockType.HEADING == "heading"


@pytest.mark.unit
class TestParseReport:
    def test_parse_report_required_fields(self):
        report = ParseReport(
            char_count=1500,
            block_count=10,
            non_printable_ratio=0.01,
            repetition_score=0.05,
            parser_used="pymupdf",
        )
        assert report.char_count == 1500
        assert report.block_count == 10
        assert report.non_printable_ratio == 0.01
        assert report.repetition_score == 0.05
        assert report.parser_used == "pymupdf"
        assert report.fallback_triggered is False

    def test_parse_report_fallback_triggered(self):
        report = ParseReport(
            char_count=100,
            block_count=2,
            non_printable_ratio=0.0,
            repetition_score=0.0,
            parser_used="trafilatura",
            fallback_triggered=True,
        )
        assert report.fallback_triggered is True

    def test_parse_report_ratio_bounds(self):
        with pytest.raises(Exception):
            ParseReport(
                char_count=0, block_count=0,
                non_printable_ratio=1.5, repetition_score=0.0,
                parser_used="x",
            )


@pytest.mark.unit
class TestDocument:
    def test_document_minimal(self):
        doc = Document(doc_id="abc123", source_path="/tmp/file.pdf")
        assert doc.doc_id == "abc123"
        assert doc.source_path == "/tmp/file.pdf"
        assert doc.mime_type == ""
        assert doc.metadata == {}
        assert doc.blocks == []
        assert doc.parse_report is None

    def test_document_with_blocks_and_report(self):
        block = IRBlock(text="Some text", block_type=BlockType.PARAGRAPH)
        report = ParseReport(
            char_count=9,
            block_count=1,
            non_printable_ratio=0.0,
            repetition_score=0.0,
            parser_used="md_parser",
        )
        doc = Document(
            doc_id="doc-001",
            source_path="/docs/readme.md",
            mime_type="text/markdown",
            metadata={"author": "Alice"},
            blocks=[block],
            parse_report=report,
        )
        assert len(doc.blocks) == 1
        assert doc.blocks[0].text == "Some text"
        assert doc.parse_report.parser_used == "md_parser"
        assert doc.metadata["author"] == "Alice"

    def test_document_importable_no_circular_deps(self):
        # If we get here, imports resolved without circular dependency errors.
        from rag.core.contracts.document import Document as D  # noqa: F401
        assert D is not None
