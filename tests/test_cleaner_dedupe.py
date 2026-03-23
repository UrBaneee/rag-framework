"""Tests for the DedupeParagraphs cleaner."""

import pytest

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.infra.cleaning.dedupe_paragraphs import DedupeParagraphs


def _block(text: str, page: int | None = None) -> IRBlock:
    return IRBlock(block_type=BlockType.PARAGRAPH, text=text, page=page)


@pytest.fixture()
def cleaner() -> DedupeParagraphs:
    return DedupeParagraphs()


def test_clean_no_duplicates_unchanged(cleaner):
    blocks = [_block("Alpha"), _block("Beta"), _block("Gamma")]
    result = cleaner.clean(blocks)
    assert [b.text for b in result] == ["Alpha", "Beta", "Gamma"]


def test_clean_removes_exact_duplicate(cleaner):
    blocks = [_block("Hello"), _block("Hello")]
    result = cleaner.clean(blocks)
    assert len(result) == 1
    assert result[0].text == "Hello"


def test_clean_keeps_first_occurrence(cleaner):
    blocks = [_block("A"), _block("B"), _block("A"), _block("C"), _block("B")]
    result = cleaner.clean(blocks)
    assert [b.text for b in result] == ["A", "B", "C"]


def test_clean_whitespace_normalised_for_dedup(cleaner):
    blocks = [_block("  Hello  "), _block("Hello")]
    result = cleaner.clean(blocks)
    assert len(result) == 1


def test_clean_empty_input(cleaner):
    assert cleaner.clean([]) == []


def test_clean_all_duplicates(cleaner):
    blocks = [_block("Same")] * 5
    result = cleaner.clean(blocks)
    assert len(result) == 1


def test_clean_preserves_order(cleaner):
    blocks = [_block("C"), _block("A"), _block("B"), _block("A")]
    result = cleaner.clean(blocks)
    assert [b.text for b in result] == ["C", "A", "B"]


def test_clean_preserves_block_metadata(cleaner):
    block = IRBlock(block_type=BlockType.HEADING, text="Title", page=1)
    result = cleaner.clean([block])
    assert result[0].block_type == BlockType.HEADING
    assert result[0].page == 1


def test_clean_single_block_kept(cleaner):
    block = _block("Only block")
    result = cleaner.clean([block])
    assert result == [block]
