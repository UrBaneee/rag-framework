"""Tests for Task 11.3 — resync engine (ResyncWindow and ResyncHybrid).

Acceptance criteria verified:
- Inserting one paragraph does NOT force all downstream chunks to re-embed
- Only chunks near the insertion are scheduled for re-embedding
- Resync stats are populated and traceable
"""

from __future__ import annotations

import pytest

from rag.core.contracts.chunk import Chunk
from rag.infra.chunking.resync_window import ResyncResult, ResyncWindow
from rag.infra.chunking.resync_hybrid import ResyncHybrid, ResyncStats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(sig: str, text: str = "", embedding: list[float] | None = None) -> Chunk:
    """Build a minimal Chunk for testing."""
    return Chunk(
        chunk_id=sig,
        doc_id="doc1",
        stable_text=text or sig,
        display_text=text or sig,
        chunk_signature=sig,
        block_hashes=[sig],
        embedding=embedding,
    )


def _chunks(sigs: list[str]) -> list[Chunk]:
    """Build a list of Chunks from signatures."""
    return [_make_chunk(s) for s in sigs]


# ---------------------------------------------------------------------------
# ResyncWindow — no changes
# ---------------------------------------------------------------------------


class TestResyncWindowNoChanges:
    """Identical old and new packs → everything reused."""

    def test_all_reused(self):
        old = _chunks(["a", "b", "c"])
        new = _chunks(["a", "b", "c"])
        result = ResyncWindow().resync(old, new)
        assert result.reused_count == 3
        assert result.reembed_count == 0

    def test_no_added_or_removed(self):
        old = _chunks(["a", "b"])
        new = _chunks(["a", "b"])
        result = ResyncWindow().resync(old, new)
        assert result.added_count == 0
        assert result.removed_count == 0

    def test_savings_ratio_one(self):
        old = _chunks(["x", "y"])
        new = _chunks(["x", "y"])
        result = ResyncWindow().resync(old, new)
        assert result.savings_ratio == 1.0

    def test_empty_inputs(self):
        result = ResyncWindow().resync([], [])
        assert result.reused_count == 0
        assert result.reembed_count == 0


# ---------------------------------------------------------------------------
# ResyncWindow — one insertion (key acceptance criterion)
# ---------------------------------------------------------------------------


class TestResyncWindowOneInsertion:
    """Inserting one paragraph affects only nearby chunks."""

    def test_one_insertion_middle(self):
        # Old: [a, b, c, d, e] → New: [a, b, NEW, c, d, e]
        old = _chunks(["a", "b", "c", "d", "e"])
        new = _chunks(["a", "b", "NEW", "c", "d", "e"])
        result = ResyncWindow(window_size=1).resync(old, new)

        # Only "b" (pos 1), "NEW" (pos 2), "c" (pos 3) in window
        reembed_sigs = {c.chunk_signature for c in result.reembed}
        assert "NEW" in reembed_sigs
        assert "b" in reembed_sigs
        assert "c" in reembed_sigs

        # "a", "d", "e" should be reused
        reused_sigs = {c.chunk_signature for c in result.reused}
        assert "a" in reused_sigs
        assert "d" in reused_sigs
        assert "e" in reused_sigs

    def test_one_insertion_reduces_reembed_vs_full(self):
        old = _chunks(["a", "b", "c", "d", "e"])
        new = _chunks(["a", "b", "NEW", "c", "d", "e"])
        result = ResyncWindow(window_size=1).resync(old, new)
        # With window=1, only 3 of 6 chunks re-embed (not all 6)
        assert result.reembed_count < result.total_new

    def test_one_insertion_start(self):
        old = _chunks(["b", "c", "d"])
        new = _chunks(["NEW", "b", "c", "d"])
        result = ResyncWindow(window_size=1).resync(old, new)
        reembed_sigs = {c.chunk_signature for c in result.reembed}
        assert "NEW" in reembed_sigs
        assert "b" in reembed_sigs
        # "c" and "d" should be reused
        reused_sigs = {c.chunk_signature for c in result.reused}
        assert "c" in reused_sigs
        assert "d" in reused_sigs

    def test_one_insertion_end(self):
        old = _chunks(["a", "b", "c"])
        new = _chunks(["a", "b", "c", "NEW"])
        result = ResyncWindow(window_size=1).resync(old, new)
        reembed_sigs = {c.chunk_signature for c in result.reembed}
        assert "NEW" in reembed_sigs
        assert "c" in reembed_sigs
        reused_sigs = {c.chunk_signature for c in result.reused}
        assert "a" in reused_sigs
        assert "b" in reused_sigs


# ---------------------------------------------------------------------------
# ResyncWindow — window_size=0
# ---------------------------------------------------------------------------


class TestResyncWindowZeroWindow:
    """window_size=0 marks only directly changed chunks."""

    def test_zero_window_only_changed(self):
        old = _chunks(["a", "b", "c", "d", "e"])
        new = _chunks(["a", "b", "NEW", "d", "e"])
        result = ResyncWindow(window_size=0).resync(old, new)
        reembed_sigs = {c.chunk_signature for c in result.reembed}
        assert reembed_sigs == {"NEW"}

    def test_zero_window_saves_more(self):
        old = _chunks(["a", "b", "c", "d", "e"])
        new = _chunks(["a", "b", "NEW", "d", "e"])
        r0 = ResyncWindow(window_size=0).resync(old, new)
        r1 = ResyncWindow(window_size=1).resync(old, new)
        assert r0.reembed_count <= r1.reembed_count


# ---------------------------------------------------------------------------
# ResyncWindow — embedding propagation
# ---------------------------------------------------------------------------


class TestResyncWindowEmbeddingPropagation:
    """Old embeddings are copied to reused chunks."""

    def test_embedding_carried_forward(self):
        vec = [0.1, 0.2, 0.3]
        old = [_make_chunk("a", embedding=vec), _make_chunk("b", embedding=vec)]
        new = [_make_chunk("a"), _make_chunk("b"), _make_chunk("NEW")]
        result = ResyncWindow(window_size=0).resync(old, new)
        reused_sigs = {c.chunk_signature: c for c in result.reused}
        assert "a" in reused_sigs
        assert reused_sigs["a"].embedding == vec

    def test_no_embedding_on_reembed(self):
        old = [_make_chunk("a", embedding=[1.0])]
        new = [_make_chunk("NEW")]
        result = ResyncWindow(window_size=0).resync(old, new)
        # NEW has no embedding from old
        assert result.reembed[0].embedding is None


# ---------------------------------------------------------------------------
# ResyncWindow — added / removed signatures
# ---------------------------------------------------------------------------


class TestResyncWindowSignatures:
    def test_added_signatures(self):
        old = _chunks(["a", "b"])
        new = _chunks(["a", "b", "c"])
        result = ResyncWindow().resync(old, new)
        assert "c" in result.added_signatures

    def test_removed_signatures(self):
        old = _chunks(["a", "b", "c"])
        new = _chunks(["a", "b"])
        result = ResyncWindow().resync(old, new)
        assert "c" in result.removed_signatures

    def test_total_new(self):
        old = _chunks(["a", "b"])
        new = _chunks(["a", "b", "c"])
        result = ResyncWindow().resync(old, new)
        assert result.total_new == 3


# ---------------------------------------------------------------------------
# ResyncHybrid — basic
# ---------------------------------------------------------------------------


class TestResyncHybrid:
    def test_returns_result_and_stats(self):
        old = _chunks(["a", "b", "c"])
        new = _chunks(["a", "b", "c"])
        result, stats = ResyncHybrid().resync(old, new)
        assert isinstance(result, ResyncResult)
        assert isinstance(stats, ResyncStats)

    def test_stats_savings_ratio_no_change(self):
        old = _chunks(["a", "b", "c"])
        new = _chunks(["a", "b", "c"])
        _, stats = ResyncHybrid().resync(old, new)
        assert stats.savings_ratio == 1.0
        assert stats.reembed_count == 0

    def test_stats_populated(self):
        old = _chunks(["a", "b", "c", "d", "e"])
        new = _chunks(["a", "b", "NEW", "d", "e"])
        _, stats = ResyncHybrid(window_size=1).resync(old, new)
        assert stats.reembed_count > 0
        assert stats.added_count == 1
        assert stats.removed_count == 1

    def test_stats_skipped_and_changed_aliases(self):
        old = _chunks(["a", "b"])
        new = _chunks(["a", "NEW"])
        _, stats = ResyncHybrid(window_size=0).resync(old, new)
        assert stats.skipped_chunks == stats.reused_count
        assert stats.changed_chunks == stats.reembed_count


class TestResyncHybridFallback:
    """When too many chunks changed, fall back to full re-embed."""

    def test_fallback_triggered(self):
        # All 3 chunks changed — ratio = 1.0 >= threshold 0.8
        old = _chunks(["a", "b", "c"])
        new = _chunks(["x", "y", "z"])
        _, stats = ResyncHybrid(fallback_threshold=0.8).resync(old, new)
        assert stats.fallback_used is True
        assert stats.reused_count == 0
        assert stats.reembed_count == 3

    def test_fallback_not_triggered_below_threshold(self):
        # Only 1/5 changed — ratio = 0.2 < threshold 0.8
        old = _chunks(["a", "b", "c", "d", "e"])
        new = _chunks(["a", "b", "NEW", "d", "e"])
        _, stats = ResyncHybrid(window_size=0, fallback_threshold=0.8).resync(old, new)
        assert stats.fallback_used is False

    def test_fallback_threshold_validation(self):
        with pytest.raises(ValueError):
            ResyncHybrid(fallback_threshold=1.5)

    def test_window_size_validation(self):
        with pytest.raises(ValueError):
            ResyncWindow(window_size=-1)
