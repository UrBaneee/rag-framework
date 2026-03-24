"""Tests for Task 12.1 — pronoun risk detection.

Acceptance criteria:
- Chunks with "it/this/they" get a non-zero risk score
- Chunks without pronouns get score 0.0
- Score is in [0.0, 1.0]
- Metadata is annotated when annotate=True
"""

from __future__ import annotations

import pytest

from rag.core.contracts.chunk import Chunk
from rag.infra.chunking.pronoun_risk_rules import PronounRiskResult, PronounRiskScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, chunk_id: str = "c1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        stable_text=text,
        display_text=text,
        chunk_signature=chunk_id,
        block_hashes=[chunk_id],
    )


# ---------------------------------------------------------------------------
# score() raw function
# ---------------------------------------------------------------------------


class TestScoreFunction:
    def test_no_pronouns_returns_zero(self):
        scorer = PronounRiskScorer()
        score, count, total, matched = scorer.score(
            "The document describes the architecture of a modular pipeline."
        )
        assert score == 0.0
        assert count == 0

    def test_pronouns_return_nonzero(self):
        scorer = PronounRiskScorer()
        score, count, total, matched = scorer.score(
            "It is important to note that this affects them significantly."
        )
        assert score > 0.0
        assert count > 0

    def test_score_bounded_zero_to_one(self):
        scorer = PronounRiskScorer()
        # Pathological text: all pronouns
        score, _, _, _ = scorer.score("it it it it it it it it it it")
        assert 0.0 <= score <= 1.0

    def test_high_density_score(self):
        scorer = PronounRiskScorer()
        score, _, _, _ = scorer.score(
            "It is they who said this to them when their work was done."
        )
        assert score >= 0.2

    def test_short_text_returns_zero(self):
        scorer = PronounRiskScorer()
        # Less than min_words (default 5)
        score, count, total, _ = scorer.score("it this they")
        assert score == 0.0

    def test_matched_pronouns_listed(self):
        scorer = PronounRiskScorer()
        _, _, _, matched = scorer.score(
            "It is important and this is relevant they said."
        )
        assert "it" in matched
        assert "this" in matched
        assert "they" in matched

    def test_case_insensitive(self):
        scorer = PronounRiskScorer()
        score_lower, _, _, _ = scorer.score(
            "it is important that this affects them significantly here."
        )
        score_upper, _, _, _ = scorer.score(
            "It Is Important That This Affects Them Significantly Here."
        )
        assert abs(score_lower - score_upper) < 1e-9


# ---------------------------------------------------------------------------
# score_chunk()
# ---------------------------------------------------------------------------


class TestScoreChunk:
    def test_returns_pronoun_risk_result(self):
        scorer = PronounRiskScorer()
        chunk = _make_chunk(
            "It is important to understand this concept and its implications here."
        )
        result = scorer.score_chunk(chunk)
        assert isinstance(result, PronounRiskResult)

    def test_chunk_id_set_on_result(self):
        scorer = PronounRiskScorer()
        chunk = _make_chunk("it is a simple sentence they all know about it.", "my-chunk")
        result = scorer.score_chunk(chunk)
        assert result.chunk_id == "my-chunk"

    def test_nonzero_score_for_pronoun_chunk(self):
        scorer = PronounRiskScorer()
        chunk = _make_chunk(
            "It is important to note that this affects them greatly in their work."
        )
        result = scorer.score_chunk(chunk)
        assert result.score > 0.0
        assert result.pronoun_count > 0

    def test_zero_score_for_clean_chunk(self):
        scorer = PronounRiskScorer()
        chunk = _make_chunk(
            "The RAG framework processes documents using a modular pipeline architecture."
        )
        result = scorer.score_chunk(chunk)
        assert result.score == 0.0

    def test_metadata_annotated_when_annotate_true(self):
        scorer = PronounRiskScorer(annotate=True)
        chunk = _make_chunk(
            "It is important to understand this concept for the pipeline."
        )
        scorer.score_chunk(chunk)
        assert "pronoun_risk" in chunk.metadata
        assert "pronoun_risk_level" in chunk.metadata
        assert isinstance(chunk.metadata["pronoun_risk"], float)

    def test_metadata_not_annotated_when_annotate_false(self):
        scorer = PronounRiskScorer(annotate=False)
        chunk = _make_chunk(
            "It is important to understand this for the pipeline architecture."
        )
        scorer.score_chunk(chunk)
        assert "pronoun_risk" not in chunk.metadata

    def test_risk_level_low_for_clean_chunk(self):
        scorer = PronounRiskScorer()
        chunk = _make_chunk(
            "The framework handles document parsing with high precision always."
        )
        result = scorer.score_chunk(chunk)
        assert result.risk_level == "low"

    def test_risk_level_high_for_dense_pronoun_chunk(self):
        scorer = PronounRiskScorer()
        # ~50% pronouns
        chunk = _make_chunk(
            "It is they who said this to them when their work was finally done here."
        )
        result = scorer.score_chunk(chunk)
        assert result.risk_level in ("medium", "high")


# ---------------------------------------------------------------------------
# score_chunks() batch
# ---------------------------------------------------------------------------


class TestScoreChunks:
    def test_returns_one_result_per_chunk(self):
        scorer = PronounRiskScorer()
        chunks = [
            _make_chunk("it is important for this system", f"c{i}")
            for i in range(5)
        ]
        results = scorer.score_chunks(chunks)
        assert len(results) == 5

    def test_empty_list(self):
        scorer = PronounRiskScorer()
        results = scorer.score_chunks([])
        assert results == []

    def test_mixed_risk_scores(self):
        scorer = PronounRiskScorer()
        clean = _make_chunk(
            "The pipeline architecture processes structured documents efficiently.", "clean"
        )
        risky = _make_chunk(
            "It is they who said this to them when their work was finally done here.", "risky"
        )
        results = scorer.score_chunks([clean, risky])
        scores = {r.chunk_id: r.score for r in results}
        assert scores["risky"] > scores["clean"]


# ---------------------------------------------------------------------------
# Custom pronoun set
# ---------------------------------------------------------------------------


class TestCustomPronounSet:
    def test_custom_pronouns_only(self):
        scorer = PronounRiskScorer(pronouns=frozenset({"foo", "bar"}))
        score, count, _, _ = scorer.score(
            "foo and bar are used here repeatedly in this sentence."
        )
        assert count == 2

    def test_default_pronouns_not_matched_with_custom_set(self):
        scorer = PronounRiskScorer(pronouns=frozenset({"xyz"}))
        score, count, _, _ = scorer.score(
            "it is important that this is done carefully for them here."
        )
        assert count == 0
        assert score == 0.0


# ---------------------------------------------------------------------------
# PronounRiskResult properties
# ---------------------------------------------------------------------------


class TestPronounRiskResult:
    def test_risk_level_low(self):
        r = PronounRiskResult(chunk_id="c", score=0.1, pronoun_count=1, total_words=20)
        assert r.risk_level == "low"

    def test_risk_level_medium(self):
        r = PronounRiskResult(chunk_id="c", score=0.3, pronoun_count=3, total_words=10)
        assert r.risk_level == "medium"

    def test_risk_level_high(self):
        r = PronounRiskResult(chunk_id="c", score=0.6, pronoun_count=6, total_words=10)
        assert r.risk_level == "high"
