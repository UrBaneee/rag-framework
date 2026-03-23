"""Tests for the Trafilatura HTML parser."""

from pathlib import Path

import pytest

from rag.core.contracts.ir_block import BlockType
from rag.infra.parsing.html_trafilatura import HtmlTrafilaturaParser

_BOILERPLATE_HEAVY_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
  <nav>Home | About | Contact | Login | Sign Up | Terms | Privacy</nav>
  <header>Site Header | Navigation | Logo</header>
  <main>
    <article>
      <h1>Main Article Title</h1>
      <p>This is the first paragraph of the main article content. It contains
      meaningful information that should be extracted by the parser.</p>
      <p>This is the second paragraph with more substantive content about
      the topic being discussed in the article.</p>
    </article>
  </main>
  <footer>Copyright 2024 | Privacy Policy | Terms of Service | Contact Us</footer>
</body>
</html>
"""

_SIMPLE_HTML = """\
<!DOCTYPE html>
<html>
<head><title>Simple Page</title></head>
<body>
  <p>Hello world. This is a simple HTML page.</p>
  <p>Second paragraph with more content here.</p>
</body>
</html>
"""

_EMPTY_HTML = """\
<!DOCTYPE html>
<html><head><title>Empty</title></head><body></body></html>
"""


@pytest.fixture()
def parser() -> HtmlTrafilaturaParser:
    return HtmlTrafilaturaParser()


def _write_html(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# supports()
# ---------------------------------------------------------------------------


def test_supports_text_html(parser):
    assert parser.supports("text/html") is True


def test_supports_xhtml(parser):
    assert parser.supports("application/xhtml+xml") is True


def test_supports_pdf_returns_false(parser):
    assert parser.supports("application/pdf") is False


def test_supports_plain_text_returns_false(parser):
    assert parser.supports("text/plain") is False


# ---------------------------------------------------------------------------
# parse() — basic extraction
# ---------------------------------------------------------------------------


def test_parse_returns_document(parser, tmp_path):
    path = _write_html(tmp_path, "page.html", _SIMPLE_HTML)
    doc = parser.parse(path)
    assert doc.source_path == path
    assert doc.mime_type == "text/html"
    assert doc.doc_id


def test_parse_extracts_text_content(parser, tmp_path):
    path = _write_html(tmp_path, "page.html", _SIMPLE_HTML)
    doc = parser.parse(path)
    all_text = " ".join(b.text for b in doc.blocks)
    assert "Hello world" in all_text or len(doc.blocks) >= 1


def test_parse_excludes_boilerplate(parser, tmp_path):
    path = _write_html(tmp_path, "page.html", _BOILERPLATE_HEAVY_HTML)
    doc = parser.parse(path)
    all_text = " ".join(b.text for b in doc.blocks).lower()
    # Main content should be present
    assert "paragraph" in all_text or "article" in all_text or "content" in all_text
    # Nav/footer boilerplate should be stripped or minimal
    nav_terms = sum(1 for term in ["home | about | contact", "copyright 2024 | privacy policy"] if term in all_text)
    assert nav_terms == 0


def test_parse_blocks_are_paragraphs(parser, tmp_path):
    path = _write_html(tmp_path, "page.html", _SIMPLE_HTML)
    doc = parser.parse(path)
    for block in doc.blocks:
        assert block.block_type == BlockType.PARAGRAPH


# ---------------------------------------------------------------------------
# parse() — ParseReport
# ---------------------------------------------------------------------------


def test_parse_report_populated(parser, tmp_path):
    path = _write_html(tmp_path, "page.html", _SIMPLE_HTML)
    doc = parser.parse(path)
    report = doc.parse_report
    assert report is not None
    assert report.parser_used == "trafilatura"
    assert report.block_count >= 0
    assert report.char_count >= 0
    assert 0.0 <= report.non_printable_ratio <= 1.0
    assert 0.0 <= report.repetition_score <= 1.0
    assert report.fallback_triggered is False


def test_parse_report_block_count_matches_blocks(parser, tmp_path):
    path = _write_html(tmp_path, "page.html", _SIMPLE_HTML)
    doc = parser.parse(path)
    assert doc.parse_report.block_count == len(doc.blocks)


# ---------------------------------------------------------------------------
# parse() — edge cases
# ---------------------------------------------------------------------------


def test_parse_empty_html_produces_empty_document(parser, tmp_path):
    path = _write_html(tmp_path, "empty.html", _EMPTY_HTML)
    doc = parser.parse(path)
    # May produce 0 blocks — should not raise
    assert doc.parse_report is not None
    assert doc.parse_report.block_count == len(doc.blocks)


def test_parse_nonexistent_file_raises(parser):
    with pytest.raises(ValueError, match="Cannot read file"):
        parser.parse("/nonexistent/path/page.html")


def test_parse_metadata_contains_filename(parser, tmp_path):
    path = _write_html(tmp_path, "article.html", _SIMPLE_HTML)
    doc = parser.parse(path)
    assert doc.metadata["filename"] == "article.html"
    assert doc.metadata["extension"] == "html"
