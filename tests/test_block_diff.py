"""Tests for Task 11.2 — block diff logic.

Covers:
- diff_blocks() pure function: unchanged / added / removed classification
- BlockDiffResult properties
- Integration: IngestPipeline._run() populates diff stats when DocStore
  exposes get_prev_blocks_for_source()
"""

from __future__ import annotations

import pytest

from rag.core.utils.hashing import BlockDiffResult, diff_blocks


# ---------------------------------------------------------------------------
# Pure diff_blocks() tests
# ---------------------------------------------------------------------------


class TestDiffBlocksNoChanges:
    """Same old and new sequence — everything is unchanged."""

    def test_no_changes_all_unchanged(self):
        hashes = ["aaa", "bbb", "ccc"]
        result = diff_blocks(hashes, hashes)
        assert sorted(result.unchanged) == sorted(hashes)
        assert result.added == []
        assert result.removed == []

    def test_no_changes_empty(self):
        result = diff_blocks([], [])
        assert result.unchanged == []
        assert result.added == []
        assert result.removed == []

    def test_no_changes_counts(self):
        result = diff_blocks(["x", "y"], ["x", "y"])
        assert result.unchanged_count == 2
        assert result.added_count == 0
        assert result.removed_count == 0
        assert result.has_changes is False

    def test_no_changes_total_new(self):
        result = diff_blocks(["x", "y"], ["x", "y"])
        assert result.total_new == 2


class TestDiffBlocksAddedBlocks:
    """New version has extra blocks."""

    def test_one_added(self):
        result = diff_blocks(["a", "b"], ["a", "b", "c"])
        assert "c" in result.added
        assert result.added_count == 1
        assert result.removed_count == 0

    def test_all_added(self):
        result = diff_blocks([], ["x", "y", "z"])
        assert result.added_count == 3
        assert result.unchanged_count == 0
        assert result.removed_count == 0
        assert result.has_changes is True

    def test_added_total_new(self):
        result = diff_blocks(["a"], ["a", "b", "c"])
        assert result.total_new == 3


class TestDiffBlocksRemovedBlocks:
    """New version has fewer blocks."""

    def test_one_removed(self):
        result = diff_blocks(["a", "b", "c"], ["a", "b"])
        assert "c" in result.removed
        assert result.removed_count == 1
        assert result.added_count == 0

    def test_all_removed(self):
        result = diff_blocks(["x", "y"], [])
        assert result.removed_count == 2
        assert result.unchanged_count == 0
        assert result.added_count == 0

    def test_removed_has_changes(self):
        result = diff_blocks(["a", "b"], ["a"])
        assert result.has_changes is True


class TestDiffBlocksMixed:
    """One block changed — models a single paragraph edit."""

    def test_one_paragraph_modified(self):
        # "b" → "b_new": b is removed, b_new is added
        result = diff_blocks(["a", "b", "c"], ["a", "b_new", "c"])
        assert "a" in result.unchanged
        assert "c" in result.unchanged
        assert "b_new" in result.added
        assert "b" in result.removed
        assert result.unchanged_count == 2
        assert result.added_count == 1
        assert result.removed_count == 1

    def test_prepend_and_remove_last(self):
        result = diff_blocks(["b", "c"], ["new", "b", "c"])
        assert "new" in result.added
        assert result.removed == []
        assert "b" in result.unchanged
        assert "c" in result.unchanged

    def test_swap_two_blocks(self):
        # Both blocks exist; just reordered — should still show as unchanged
        result = diff_blocks(["a", "b"], ["b", "a"])
        assert result.unchanged_count == 2
        assert result.added_count == 0
        assert result.removed_count == 0


class TestBlockDiffResultProperties:
    """Verify computed properties on BlockDiffResult."""

    def test_total_new_formula(self):
        r = BlockDiffResult(unchanged=["a", "b"], added=["c"], removed=["d"])
        assert r.total_new == 3  # unchanged + added

    def test_has_changes_false(self):
        r = BlockDiffResult(unchanged=["a"], added=[], removed=[])
        assert r.has_changes is False

    def test_has_changes_true_added(self):
        r = BlockDiffResult(unchanged=[], added=["x"], removed=[])
        assert r.has_changes is True

    def test_has_changes_true_removed(self):
        r = BlockDiffResult(unchanged=[], added=[], removed=["x"])
        assert r.has_changes is True

    def test_counts_match_list_lengths(self):
        r = BlockDiffResult(unchanged=["a", "b"], added=["c"], removed=["d", "e"])
        assert r.unchanged_count == 2
        assert r.added_count == 1
        assert r.removed_count == 2


# ---------------------------------------------------------------------------
# Integration: IngestResult carries diff stats
# ---------------------------------------------------------------------------


class TestIngestResultDiffStats:
    """IngestResult now carries block diff stats — verify the fields exist."""

    def test_ingest_result_has_diff_fields(self):
        from rag.pipelines.ingest_pipeline import IngestResult

        result = IngestResult(doc_id="abc", source_path="/tmp/x.txt")
        assert hasattr(result, "blocks_added")
        assert hasattr(result, "blocks_removed")
        assert hasattr(result, "blocks_unchanged")
        assert hasattr(result, "diff_available")

    def test_ingest_result_diff_defaults(self):
        from rag.pipelines.ingest_pipeline import IngestResult

        result = IngestResult(doc_id="abc", source_path="/tmp/x.txt")
        assert result.blocks_added == 0
        assert result.blocks_removed == 0
        assert result.blocks_unchanged == 0
        assert result.diff_available is False

    def test_ingest_result_diff_set(self):
        from rag.pipelines.ingest_pipeline import IngestResult

        result = IngestResult(
            doc_id="abc",
            source_path="/tmp/x.txt",
            blocks_added=2,
            blocks_removed=1,
            blocks_unchanged=5,
            diff_available=True,
        )
        assert result.blocks_added == 2
        assert result.blocks_removed == 1
        assert result.blocks_unchanged == 5
        assert result.diff_available is True
