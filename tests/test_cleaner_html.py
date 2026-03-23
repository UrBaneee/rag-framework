"""Tests for the HtmlNavFooterRemover cleaner."""

import pytest

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.infra.cleaning.html_nav_footer_remove import HtmlNavFooterRemover


def _block(text: str) -> IRBlock:
    return IRBlock(block_type=BlockType.PARAGRAPH, text=text)


@pytest.fixture()
def cleaner() -> HtmlNavFooterRemover:
    return HtmlNavFooterRemover()


# ---------------------------------------------------------------------------
# Keyword-based removal
# ---------------------------------------------------------------------------


def test_removes_privacy_policy(cleaner):
    result = cleaner.clean([_block("Privacy Policy | Terms of Service")])
    assert result == []


def test_removes_copyright(cleaner):
    result = cleaner.clean([_block("© 2024 Acme Corp. All rights reserved.")])
    assert result == []


def test_removes_skip_to_content(cleaner):
    result = cleaner.clean([_block("Skip to content")])
    assert result == []


def test_removes_back_to_top(cleaner):
    result = cleaner.clean([_block("Back to top")])
    assert result == []


def test_removes_cookie_policy(cleaner):
    result = cleaner.clean([_block("Cookie policy: we use cookies.")])
    assert result == []


def test_removes_all_rights_reserved(cleaner):
    result = cleaner.clean([_block("All rights reserved 2024.")])
    assert result == []


# ---------------------------------------------------------------------------
# Nav-list structural detection
# ---------------------------------------------------------------------------


def test_removes_pipe_separated_nav(cleaner):
    result = cleaner.clean([_block("Home | About | Services | Contact | Login")])
    assert result == []


def test_removes_bullet_separated_nav(cleaner):
    result = cleaner.clean([_block("Home • Products • Support • Blog")])
    assert result == []


# ---------------------------------------------------------------------------
# Content blocks are kept
# ---------------------------------------------------------------------------


def test_keeps_article_content(cleaner):
    block = _block("This is a genuine article paragraph about machine learning.")
    result = cleaner.clean([block])
    assert result == [block]


def test_keeps_heading_block(cleaner):
    block = IRBlock(block_type=BlockType.HEADING, text="Introduction to Neural Networks")
    result = cleaner.clean([block])
    assert result == [block]


def test_keeps_long_sentence(cleaner):
    text = "The quick brown fox jumps over the lazy dog, demonstrating the full alphabet."
    block = _block(text)
    result = cleaner.clean([block])
    assert result == [block]


# ---------------------------------------------------------------------------
# Mixed list
# ---------------------------------------------------------------------------


def test_filters_mixed_list(cleaner):
    blocks = [
        _block("Home | About | Contact"),
        _block("Real article content here."),
        _block("© 2024 Company"),
        _block("Another paragraph of useful text."),
    ]
    result = cleaner.clean(blocks)
    assert len(result) == 2
    assert result[0].text == "Real article content here."
    assert result[1].text == "Another paragraph of useful text."


def test_empty_input(cleaner):
    assert cleaner.clean([]) == []
