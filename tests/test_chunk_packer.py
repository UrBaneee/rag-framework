"""Tests for the AnchorAwareChunkPacker."""

import pytest

from rag.core.contracts.ir_block import BlockType
from rag.core.contracts.text_block import TextBlock
from rag.infra.chunking.chunk_packer_anchor_aware import (
    AnchorAwareChunkPacker,
    _approx_tokens,
    _compute_chunk_signature,
)

DOC_ID = "doc_001"
_FAKE_HASH = "a" * 64


def _tb(
    text: str,
    block_type: BlockType = BlockType.PARAGRAPH,
    sequence: int = 0,
    page: int | None = None,
    block_hash: str | None = None,
) -> TextBlock:
    return TextBlock(
        doc_id=DOC_ID,
        block_type=block_type,
        text=text,
        block_hash=block_hash or _FAKE_HASH,
        sequence=sequence,
        page=page,
    )


def _seq_blocks(texts: list[str], block_type: BlockType = BlockType.PARAGRAPH) -> list[TextBlock]:
    return [
        _tb(text, block_type=block_type, sequence=i, block_hash=f"{i:064x}")
        for i, text in enumerate(texts)
    ]


@pytest.fixture()
def packer() -> AnchorAwareChunkPacker:
    return AnchorAwareChunkPacker(token_budget=512)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def test_approx_tokens_non_zero():
    assert _approx_tokens("Hello world.") >= 1


def test_approx_tokens_empty_returns_one():
    assert _approx_tokens("") == 1


def test_chunk_signature_64_chars():
    sig = _compute_chunk_signature(["abc", "def"])
    assert len(sig) == 64


def test_chunk_signature_stable():
    assert _compute_chunk_signature(["a", "b"]) == _compute_chunk_signature(["a", "b"])


def test_chunk_signature_order_dependent():
    assert _compute_chunk_signature(["a", "b"]) != _compute_chunk_signature(["b", "a"])


# ---------------------------------------------------------------------------
# pack() — basic behaviour
# ---------------------------------------------------------------------------


def test_pack_empty_input(packer):
    assert packer.pack([]) == []


def test_pack_single_block_produces_one_chunk(packer):
    blocks = _seq_blocks(["Single paragraph."])
    result = packer.pack(blocks)
    assert len(result) == 1


def test_pack_doc_id_propagated(packer):
    blocks = _seq_blocks(["Content."])
    result = packer.pack(blocks)
    assert result[0].doc_id == DOC_ID


def test_pack_stable_text_contains_block_text(packer):
    blocks = _seq_blocks(["Hello world."])
    result = packer.pack(blocks)
    assert "Hello world." in result[0].stable_text


def test_pack_block_hashes_correct(packer):
    blocks = _seq_blocks(["Alpha.", "Beta."])
    result = packer.pack(blocks)
    all_hashes = [h for chunk in result for h in chunk.block_hashes]
    assert blocks[0].block_hash in all_hashes
    assert blocks[1].block_hash in all_hashes


def test_pack_chunk_signature_computed(packer):
    blocks = _seq_blocks(["Alpha.", "Beta."])
    result = packer.pack(blocks)
    for chunk in result:
        assert len(chunk.chunk_signature) == 64


def test_pack_loc_span_present(packer):
    blocks = _seq_blocks(["A.", "B.", "C."])
    result = packer.pack(blocks)
    for chunk in result:
        assert "loc_span" in chunk.metadata
        span = chunk.metadata["loc_span"]
        assert len(span) == 2
        assert span[0] <= span[1]


# ---------------------------------------------------------------------------
# pack() — anchor boundary splitting
# ---------------------------------------------------------------------------


def test_pack_splits_at_heading(packer):
    blocks = [
        _tb("Introduction", block_type=BlockType.HEADING, sequence=0, block_hash="0" * 64),
        _tb("First paragraph.", sequence=1, block_hash="1" * 64),
        _tb("Second paragraph.", sequence=2, block_hash="2" * 64),
        _tb("New Section", block_type=BlockType.HEADING, sequence=3, block_hash="3" * 64),
        _tb("Section content.", sequence=4, block_hash="4" * 64),
    ]
    result = packer.pack(blocks)
    # Should split at "New Section" heading → at least 2 chunks
    assert len(result) >= 2


def test_pack_all_paragraphs_may_fit_in_one_chunk():
    packer = AnchorAwareChunkPacker(token_budget=1000)
    blocks = _seq_blocks(["Short.", "Also short.", "And this too."])
    result = packer.pack(blocks)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# pack() — token budget splitting
# ---------------------------------------------------------------------------


def test_pack_splits_when_budget_exceeded():
    packer = AnchorAwareChunkPacker(token_budget=5)  # very small budget
    long_text = "A" * 100  # ≈ 25 tokens
    blocks = _seq_blocks([long_text, long_text, long_text])
    result = packer.pack(blocks)
    assert len(result) >= 2


# ---------------------------------------------------------------------------
# pack() — metadata
# ---------------------------------------------------------------------------


def test_pack_page_numbers_in_metadata(packer):
    blocks = [
        _tb("Content page 1.", page=1, sequence=0, block_hash="0" * 64),
        _tb("Content page 2.", page=2, sequence=1, block_hash="1" * 64),
    ]
    result = packer.pack(blocks)
    assert result[0].metadata.get("start_page") == 1


def test_pack_token_count_positive(packer):
    blocks = _seq_blocks(["Some content here."])
    result = packer.pack(blocks)
    assert result[0].token_count > 0


def test_pack_display_text_present(packer):
    blocks = _seq_blocks(["Plain paragraph."])
    result = packer.pack(blocks)
    assert result[0].display_text


def test_pack_chunk_signature_stable_across_calls(packer):
    blocks = _seq_blocks(["Alpha.", "Beta."])
    r1 = packer.pack(blocks)
    r2 = packer.pack(blocks)
    assert r1[0].chunk_signature == r2[0].chunk_signature
