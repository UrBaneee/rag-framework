"""Tests for the ParagraphBlockSplitter."""

import pytest

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.infra.chunking.block_splitter_paragraph import (
    ParagraphBlockSplitter,
    _compute_block_hash,
)

DOC_ID = "test_doc_001"


def _ir(text: str, block_type: BlockType = BlockType.PARAGRAPH, page: int | None = None, section_path: list[str] | None = None) -> IRBlock:
    return IRBlock(
        block_type=block_type,
        text=text,
        page=page,
        section_path=section_path or [],
    )


@pytest.fixture()
def splitter() -> ParagraphBlockSplitter:
    return ParagraphBlockSplitter()


# ---------------------------------------------------------------------------
# _compute_block_hash
# ---------------------------------------------------------------------------


def test_hash_is_64_hex_chars():
    h = _compute_block_hash("Hello world.")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_is_stable():
    assert _compute_block_hash("Hello") == _compute_block_hash("Hello")


def test_hash_differs_for_different_text():
    assert _compute_block_hash("Alpha") != _compute_block_hash("Beta")


def test_hash_strips_whitespace():
    assert _compute_block_hash("  Hello  ") == _compute_block_hash("Hello")


def test_hash_collapses_internal_whitespace():
    assert _compute_block_hash("Hello  world") == _compute_block_hash("Hello world")


def test_hash_nfc_normalisation():
    # e + combining acute == é (NFC)
    assert _compute_block_hash("e\u0301") == _compute_block_hash("\u00e9")


# ---------------------------------------------------------------------------
# split() — basic behaviour
# ---------------------------------------------------------------------------


def test_split_five_paragraphs_produces_five_blocks(splitter):
    blocks = [_ir(f"Paragraph {i}.") for i in range(5)]
    result = splitter.split(DOC_ID, blocks)
    assert len(result) == 5


def test_split_assigns_doc_id(splitter):
    result = splitter.split(DOC_ID, [_ir("Hello.")])
    assert result[0].doc_id == DOC_ID


def test_split_sequence_starts_at_zero(splitter):
    blocks = [_ir(f"Block {i}.") for i in range(3)]
    result = splitter.split(DOC_ID, blocks)
    assert [b.sequence for b in result] == [0, 1, 2]


def test_split_sequence_is_contiguous(splitter):
    blocks = [_ir(f"Para {i}.") for i in range(5)]
    result = splitter.split(DOC_ID, blocks)
    sequences = [b.sequence for b in result]
    assert sequences == list(range(5))


def test_split_preserves_order(splitter):
    blocks = [_ir(f"Block {i}.") for i in range(5)]
    result = splitter.split(DOC_ID, blocks)
    for i, tb in enumerate(result):
        assert f"Block {i}." in tb.text


def test_split_computes_block_hash(splitter):
    block = _ir("Hello world.")
    result = splitter.split(DOC_ID, [block])
    assert result[0].block_hash == _compute_block_hash("Hello world.")


def test_split_hashes_are_stable_across_calls(splitter):
    blocks = [_ir("Paragraph text.")]
    r1 = splitter.split(DOC_ID, blocks)
    r2 = splitter.split(DOC_ID, blocks)
    assert r1[0].block_hash == r2[0].block_hash


# ---------------------------------------------------------------------------
# split() — metadata propagation
# ---------------------------------------------------------------------------


def test_split_preserves_block_type(splitter):
    block = _ir("Title", block_type=BlockType.HEADING)
    result = splitter.split(DOC_ID, [block])
    assert result[0].block_type == BlockType.HEADING


def test_split_preserves_page_number(splitter):
    block = _ir("Content.", page=3)
    result = splitter.split(DOC_ID, [block])
    assert result[0].page == 3


def test_split_preserves_section_path(splitter):
    block = _ir("Content.", section_path=["Chapter 1", "Intro"])
    result = splitter.split(DOC_ID, [block])
    assert result[0].section_path == ["Chapter 1", "Intro"]


# ---------------------------------------------------------------------------
# split() — edge cases
# ---------------------------------------------------------------------------


def test_split_empty_input(splitter):
    assert splitter.split(DOC_ID, []) == []


def test_split_skips_empty_blocks(splitter):
    blocks = [_ir("Real."), _ir(""), _ir("   "), _ir("Also real.")]
    result = splitter.split(DOC_ID, blocks)
    assert len(result) == 2
    assert result[0].sequence == 0
    assert result[1].sequence == 1


def test_split_block_id_is_none(splitter):
    result = splitter.split(DOC_ID, [_ir("Hello.")])
    assert result[0].block_id is None
