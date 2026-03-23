"""Tests for LightContextPacker — Task 7.3."""

import pytest

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.infra.generation.context_packer_light import LightContextPacker, PackedContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    chunk_id: str,
    display_text: str,
    stable_text: str | None = None,
    doc_id: str = "doc1",
    final_score: float = 1.0,
    metadata: dict | None = None,
) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id=doc_id,
        display_text=display_text,
        stable_text=stable_text if stable_text is not None else display_text,
        rrf_score=final_score,
        final_score=final_score,
        retrieval_source=RetrievalSource.BM25,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Basic packing
# ---------------------------------------------------------------------------


def test_pack_returns_packed_context():
    packer = LightContextPacker(top_k=3)
    cands = [_make_candidate(f"c{i}", f"Chunk text {i}") for i in range(5)]
    result = packer.pack(cands)
    assert isinstance(result, PackedContext)


def test_pack_limits_to_top_k():
    packer = LightContextPacker(top_k=3)
    cands = [_make_candidate(f"c{i}", f"Chunk text {i}") for i in range(5)]
    result = packer.pack(cands)
    assert len(result.candidates) == 3


def test_pack_fewer_than_top_k_returns_all():
    packer = LightContextPacker(top_k=5)
    cands = [_make_candidate(f"c{i}", f"Chunk text {i}") for i in range(3)]
    result = packer.pack(cands)
    assert len(result.candidates) == 3


def test_pack_preserves_order():
    packer = LightContextPacker(top_k=3)
    cands = [_make_candidate(f"c{i}", f"Chunk text {i}") for i in range(5)]
    result = packer.pack(cands)
    ids = [c.chunk_id for c in result.candidates]
    assert ids == ["c0", "c1", "c2"]


def test_pack_empty_candidates():
    packer = LightContextPacker(top_k=3)
    result = packer.pack([])
    assert result.candidates == []
    assert result.citations == []
    assert result.context_text == ""
    assert result.total_tokens == 0


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def test_pack_deduplicates_same_stable_text():
    packer = LightContextPacker(top_k=3)
    cands = [
        _make_candidate("c0", "Chunk A display", stable_text="unique text alpha"),
        _make_candidate("c1", "Chunk B display", stable_text="duplicate text"),
        _make_candidate("c2", "Chunk C display", stable_text="duplicate text"),  # dup
        _make_candidate("c3", "Chunk D display", stable_text="unique text beta"),
        _make_candidate("c4", "Chunk E display", stable_text="unique text gamma"),
    ]
    result = packer.pack(cands)
    stable_texts = [c.stable_text for c in result.candidates]
    assert stable_texts.count("duplicate text") == 1
    assert len(result.candidates) == 3


def test_pack_skips_whitespace_only_duplicates():
    packer = LightContextPacker(top_k=3)
    # Two candidates with same stable_text but different leading whitespace
    cands = [
        _make_candidate("c0", "Text A", stable_text="  same text  "),
        _make_candidate("c1", "Text B", stable_text="same text"),  # same after strip
        _make_candidate("c2", "Text C", stable_text="different text"),
    ]
    result = packer.pack(cands)
    assert len(result.candidates) == 2
    assert result.candidates[0].chunk_id == "c0"
    assert result.candidates[1].chunk_id == "c2"


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def test_pack_respects_token_budget():
    # Each chunk: display_text has 80 chars → ~20 tokens; budget = 45 → fits 2
    packer = LightContextPacker(top_k=5, token_budget=45)
    text = "A" * 80  # 80 chars → 20 tokens each
    cands = [_make_candidate(f"c{i}", text, stable_text=f"unique {i}") for i in range(5)]
    result = packer.pack(cands)
    assert len(result.candidates) == 2
    assert result.truncated is True


def test_pack_not_truncated_when_within_budget():
    packer = LightContextPacker(top_k=3, token_budget=1000)
    cands = [_make_candidate(f"c{i}", "short") for i in range(3)]
    result = packer.pack(cands)
    assert result.truncated is False


def test_pack_uses_metadata_token_count():
    packer = LightContextPacker(top_k=5, token_budget=50)
    # metadata token_count=30 per chunk; budget=50 → only 1 fits
    cands = [
        _make_candidate(f"c{i}", "x" * 10, stable_text=f"u{i}", metadata={"token_count": 30})
        for i in range(5)
    ]
    result = packer.pack(cands)
    assert len(result.candidates) == 1
    assert result.truncated is True


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_pack_produces_citation_mapping():
    packer = LightContextPacker(top_k=3)
    cands = [_make_candidate(f"c{i}", f"Text {i}") for i in range(3)]
    result = packer.pack(cands)
    assert len(result.citations) == len(result.candidates)


def test_citations_are_one_based():
    packer = LightContextPacker(top_k=3)
    cands = [_make_candidate(f"c{i}", f"Text {i}") for i in range(3)]
    result = packer.pack(cands)
    ref_numbers = [c.ref_number for c in result.citations]
    assert ref_numbers == [1, 2, 3]


def test_citation_chunk_ids_match_candidates():
    packer = LightContextPacker(top_k=3)
    cands = [_make_candidate(f"chunk_{i}", f"Text {i}") for i in range(3)]
    result = packer.pack(cands)
    for citation, candidate in zip(result.citations, result.candidates):
        assert citation.chunk_id == candidate.chunk_id


def test_citation_source_label_uses_metadata():
    packer = LightContextPacker(top_k=1)
    cands = [_make_candidate("c0", "Text", metadata={"source_label": "report.pdf — page 5"})]
    result = packer.pack(cands)
    assert result.citations[0].source_label == "report.pdf — page 5"


def test_citation_source_label_fallback_with_page():
    packer = LightContextPacker(top_k=1)
    cands = [_make_candidate("c0", "Text", doc_id="guide.pdf", metadata={"start_page": 7})]
    result = packer.pack(cands)
    label = result.citations[0].source_label
    assert "guide.pdf" in label
    assert "7" in label


# ---------------------------------------------------------------------------
# Context text format
# ---------------------------------------------------------------------------


def test_context_text_contains_numbered_entries():
    packer = LightContextPacker(top_k=2)
    cands = [
        _make_candidate("c0", "First chunk content"),
        _make_candidate("c1", "Second chunk content"),
    ]
    result = packer.pack(cands)
    assert "[1]" in result.context_text
    assert "[2]" in result.context_text
    assert "First chunk content" in result.context_text
    assert "Second chunk content" in result.context_text


def test_context_text_empty_for_no_candidates():
    packer = LightContextPacker(top_k=3)
    result = packer.pack([])
    assert result.context_text == ""


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_top_k_raises():
    with pytest.raises(ValueError):
        LightContextPacker(top_k=0)


def test_invalid_token_budget_raises():
    with pytest.raises(ValueError):
        LightContextPacker(token_budget=0)
