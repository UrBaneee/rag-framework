"""Tests for the OcrLineMerger cleaner."""

import pytest

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.infra.cleaning.ocr_line_merge import OcrLineMerger, _ends_sentence, _merge_lines


def _block(text: str) -> IRBlock:
    return IRBlock(block_type=BlockType.PARAGRAPH, text=text)


@pytest.fixture()
def cleaner() -> OcrLineMerger:
    return OcrLineMerger(short_line_threshold=80)


# ---------------------------------------------------------------------------
# _ends_sentence helper
# ---------------------------------------------------------------------------


def test_ends_sentence_period():
    assert _ends_sentence("Hello world.") is True


def test_ends_sentence_exclamation():
    assert _ends_sentence("Stop!") is True


def test_ends_sentence_question():
    assert _ends_sentence("Really?") is True


def test_ends_sentence_no_punctuation():
    assert _ends_sentence("incomplete line") is False


def test_ends_sentence_empty():
    assert _ends_sentence("") is False


# ---------------------------------------------------------------------------
# _merge_lines helper
# ---------------------------------------------------------------------------


def test_merge_lines_single_line_unchanged():
    text = "Single complete line."
    assert _merge_lines(text, 80) == text


def test_merge_lines_joins_short_fragments():
    text = "This is a\nshort line\nthat should merge."
    result = _merge_lines(text, 80)
    assert "\n" not in result
    assert "This is a short line that should merge." == result


def test_merge_lines_keeps_sentence_boundaries():
    text = "First sentence.\nSecond sentence starts here."
    result = _merge_lines(text, 80)
    # Both lines end sentences or are long — should be separate
    assert "First sentence." in result
    assert "Second sentence starts here." in result


# ---------------------------------------------------------------------------
# OcrLineMerger.clean()
# ---------------------------------------------------------------------------


def test_clean_merges_fragmented_lines(cleaner):
    text = "The quick brown\nfox jumps over\nthe lazy dog."
    block = _block(text)
    result = cleaner.clean([block])
    assert "\n" not in result[0].text
    assert "quick brown fox jumps over" in result[0].text


def test_clean_single_line_block_unchanged(cleaner):
    block = _block("A complete single-line paragraph.")
    result = cleaner.clean([block])
    assert result[0] is block


def test_clean_empty_input(cleaner):
    assert cleaner.clean([]) == []


def test_clean_preserves_block_count(cleaner):
    blocks = [
        _block("Line one\nline two\nline three."),
        _block("Another block\nwith fragments."),
    ]
    result = cleaner.clean(blocks)
    assert len(result) == 2


def test_clean_preserves_block_type(cleaner):
    block = IRBlock(block_type=BlockType.HEADING, text="Short\nheading text.")
    result = cleaner.clean([block])
    assert result[0].block_type == BlockType.HEADING


def test_clean_preserves_page_number(cleaner):
    block = IRBlock(block_type=BlockType.PARAGRAPH, text="line one\nline two.", page=5)
    result = cleaner.clean([block])
    assert result[0].page == 5


def test_clean_custom_threshold():
    cleaner = OcrLineMerger(short_line_threshold=20)
    # Lines under 20 chars without sentence endings → merged
    text = "Short line\nand more."
    block = _block(text)
    result = cleaner.clean([block])
    assert "\n" not in result[0].text


def test_clean_long_lines_not_merged(cleaner):
    # Lines >= 80 chars are treated as complete and not merged with next
    long_line = "A" * 85
    text = f"{long_line}\nNext line here."
    block = _block(text)
    result = cleaner.clean([block])
    # Long line is flushed as a sentence boundary — new line starts separately
    assert long_line in result[0].text
