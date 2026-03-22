"""Tests for the TXT/Markdown parser."""

import textwrap
from pathlib import Path

import pytest

from rag.core.contracts.ir_block import BlockType
from rag.infra.parsing.md_parser import MdParser, _compute_non_printable_ratio, _compute_repetition_score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parser() -> MdParser:
    return MdParser()


def _write_file(tmp_path: Path, name: str, content: str) -> str:
    """Write content to a temp file and return the path as a string."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# supports()
# ---------------------------------------------------------------------------


def test_supports_text_plain(parser):
    assert parser.supports("text/plain") is True


def test_supports_text_markdown(parser):
    assert parser.supports("text/markdown") is True


def test_supports_text_x_markdown(parser):
    assert parser.supports("text/x-markdown") is True


def test_supports_pdf_returns_false(parser):
    assert parser.supports("application/pdf") is False


def test_supports_html_returns_false(parser):
    assert parser.supports("text/html") is False


# ---------------------------------------------------------------------------
# parse() — basic plain text
# ---------------------------------------------------------------------------


def test_parse_plain_text_produces_document(parser, tmp_path):
    content = "Hello world.\n\nSecond paragraph."
    path = _write_file(tmp_path, "sample.txt", content)
    doc = parser.parse(path)
    assert doc.source_path == path
    assert doc.mime_type == "text/plain"
    assert doc.doc_id  # non-empty


def test_parse_plain_text_two_paragraphs(parser, tmp_path):
    content = "First paragraph.\n\nSecond paragraph."
    path = _write_file(tmp_path, "sample.txt", content)
    doc = parser.parse(path)
    paragraphs = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH]
    assert len(paragraphs) == 2
    assert "First" in paragraphs[0].text
    assert "Second" in paragraphs[1].text


def test_parse_empty_file_produces_no_blocks(parser, tmp_path):
    path = _write_file(tmp_path, "empty.txt", "")
    doc = parser.parse(path)
    assert doc.blocks == []
    assert doc.parse_report.block_count == 0
    assert doc.parse_report.char_count == 0


# ---------------------------------------------------------------------------
# parse() — Markdown headings
# ---------------------------------------------------------------------------


def test_parse_markdown_headings(parser, tmp_path):
    content = textwrap.dedent("""\
        # Title

        Introduction paragraph.

        ## Section One

        Content under section one.
    """)
    path = _write_file(tmp_path, "doc.md", content)
    doc = parser.parse(path)
    headings = [b for b in doc.blocks if b.block_type == BlockType.HEADING]
    assert len(headings) == 2
    assert headings[0].text == "Title"
    assert headings[1].text == "Section One"


def test_parse_markdown_mime_type(parser, tmp_path):
    path = _write_file(tmp_path, "readme.md", "# Hello")
    doc = parser.parse(path)
    assert doc.mime_type == "text/markdown"


def test_parse_markdown_section_path_propagated(parser, tmp_path):
    content = textwrap.dedent("""\
        # Chapter

        Paragraph under chapter.
    """)
    path = _write_file(tmp_path, "doc.md", content)
    doc = parser.parse(path)
    para = next(b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH)
    assert "Chapter" in para.section_path


# ---------------------------------------------------------------------------
# parse() — fenced code blocks
# ---------------------------------------------------------------------------


def test_parse_fenced_code_block(parser, tmp_path):
    content = textwrap.dedent("""\
        Some intro.

        ```
        x = 1 + 2
        print(x)
        ```

        After code.
    """)
    path = _write_file(tmp_path, "code.md", content)
    doc = parser.parse(path)
    code_blocks = [b for b in doc.blocks if b.block_type == BlockType.CODE]
    assert len(code_blocks) == 1
    assert "x = 1 + 2" in code_blocks[0].text


# ---------------------------------------------------------------------------
# parse() — ParseReport fields
# ---------------------------------------------------------------------------


def test_parse_report_populated(parser, tmp_path):
    content = "Hello world.\n\nAnother paragraph."
    path = _write_file(tmp_path, "sample.txt", content)
    doc = parser.parse(path)
    report = doc.parse_report
    assert report is not None
    assert report.parser_used == "md_parser"
    assert report.block_count == 2
    assert report.char_count > 0
    assert 0.0 <= report.non_printable_ratio <= 1.0
    assert 0.0 <= report.repetition_score <= 1.0
    assert report.fallback_triggered is False


# ---------------------------------------------------------------------------
# parse() — error handling
# ---------------------------------------------------------------------------


def test_parse_nonexistent_file_raises(parser):
    with pytest.raises(ValueError, match="Cannot read file"):
        parser.parse("/nonexistent/path/file.txt")


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


def test_non_printable_ratio_clean_text():
    assert _compute_non_printable_ratio("Hello, World!") == 0.0


def test_non_printable_ratio_with_null_bytes():
    text = "Hello\x00World"
    ratio = _compute_non_printable_ratio(text)
    assert ratio > 0.0


def test_non_printable_ratio_empty_string():
    assert _compute_non_printable_ratio("") == 0.0


def test_repetition_score_no_duplicates():
    from rag.core.contracts.ir_block import IRBlock

    blocks = [
        IRBlock(block_type=BlockType.PARAGRAPH, text="Alpha"),
        IRBlock(block_type=BlockType.PARAGRAPH, text="Beta"),
    ]
    assert _compute_repetition_score(blocks) == 0.0


def test_repetition_score_with_duplicate():
    from rag.core.contracts.ir_block import IRBlock

    blocks = [
        IRBlock(block_type=BlockType.PARAGRAPH, text="Same"),
        IRBlock(block_type=BlockType.PARAGRAPH, text="Same"),
    ]
    score = _compute_repetition_score(blocks)
    assert score == 0.5


def test_repetition_score_single_block():
    from rag.core.contracts.ir_block import IRBlock

    blocks = [IRBlock(block_type=BlockType.PARAGRAPH, text="Only")]
    assert _compute_repetition_score(blocks) == 0.0
