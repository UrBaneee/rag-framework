"""Tests for the EmptyBlockFilter cleaner."""

import pytest

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.infra.cleaning.empty_filter import EmptyBlockFilter


def _block(text: str) -> IRBlock:
    return IRBlock(block_type=BlockType.PARAGRAPH, text=text)


@pytest.fixture()
def cleaner() -> EmptyBlockFilter:
    return EmptyBlockFilter()


# ---------------------------------------------------------------------------
# EmptyBlockFilter.clean() — default threshold (min_chars=1)
# ---------------------------------------------------------------------------


def test_clean_removes_empty_string(cleaner):
    result = cleaner.clean([_block("")])
    assert result == []


def test_clean_removes_whitespace_only(cleaner):
    result = cleaner.clean([_block("   \n\t  ")])
    assert result == []


def test_clean_keeps_non_empty_block(cleaner):
    block = _block("Hello world.")
    result = cleaner.clean([block])
    assert len(result) == 1
    assert result[0] is block


def test_clean_filters_mixed_list(cleaner):
    blocks = [
        _block("Real content here."),
        _block(""),
        _block("More content."),
        _block("   "),
    ]
    result = cleaner.clean(blocks)
    assert len(result) == 2
    assert result[0].text == "Real content here."
    assert result[1].text == "More content."


def test_clean_empty_input(cleaner):
    assert cleaner.clean([]) == []


def test_clean_all_empty(cleaner):
    blocks = [_block(""), _block("  "), _block("\n")]
    result = cleaner.clean(blocks)
    assert result == []


def test_clean_preserves_order(cleaner):
    blocks = [_block("A"), _block(""), _block("B"), _block("C")]
    result = cleaner.clean(blocks)
    assert [b.text for b in result] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# EmptyBlockFilter — custom min_chars threshold
# ---------------------------------------------------------------------------


def test_custom_threshold_removes_short_blocks():
    cleaner = EmptyBlockFilter(min_chars=10)
    blocks = [_block("Short"), _block("Long enough text here.")]
    result = cleaner.clean(blocks)
    assert len(result) == 1
    assert result[0].text == "Long enough text here."


def test_custom_threshold_zero_keeps_all():
    cleaner = EmptyBlockFilter(min_chars=0)
    blocks = [_block(""), _block("text")]
    result = cleaner.clean(blocks)
    # min_chars=0: all blocks with len("".strip()) >= 0 → all kept
    assert len(result) == 2


def test_custom_threshold_single_char_kept():
    cleaner = EmptyBlockFilter(min_chars=1)
    result = cleaner.clean([_block("x")])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Preserves block metadata
# ---------------------------------------------------------------------------


def test_clean_preserves_block_type():
    cleaner = EmptyBlockFilter()
    block = IRBlock(block_type=BlockType.HEADING, text="Title")
    result = cleaner.clean([block])
    assert result[0].block_type == BlockType.HEADING


def test_clean_preserves_page_number():
    cleaner = EmptyBlockFilter()
    block = IRBlock(block_type=BlockType.PARAGRAPH, text="Content", page=3)
    result = cleaner.clean([block])
    assert result[0].page == 3
