"""Deterministic hashing utilities for incremental ingestion — Tasks 11.1–11.2.

All identity functions live here so block hashes, chunk signatures, and
document fingerprints are computed consistently across every pipeline stage.

**Critical rule:** ``canonicalize()`` is the single shared implementation.
Do not inline or re-implement it elsewhere — any deviation will produce a
different hash for the same content across runs, silently breaking
incremental updates.

**Case normalization** (e.g. ``.lower()``) is opt-in via the caller, not
applied by default, to preserve semantic differences between versions.

Usage::

    from rag.core.utils.hashing import (
        canonicalize,
        file_fingerprint,
        block_hash,
        chunk_signature,
        make_doc_id,
        diff_blocks,
    )

    fp = file_fingerprint("/path/to/doc.pdf")
    doc_id = make_doc_id("/path/to/doc.pdf", fp)
    result = diff_blocks(old_hashes, new_hashes)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Text canonicalization
# ---------------------------------------------------------------------------


def canonicalize(text: str) -> str:
    """Collapse internal whitespace and strip leading/trailing whitespace.

    This is the single canonical form used for block hashing.  Preserves
    original casing — apply ``.lower()`` before calling if case-insensitive
    hashing is needed.

    Args:
        text: Raw or cleaned text string.

    Returns:
        Whitespace-normalised string with no leading/trailing spaces.

    Examples:
        >>> canonicalize("  hello   world  ")
        'hello world'
        >>> canonicalize("Line1\\n\\nLine2")
        'Line1 Line2'
    """
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# Per-block hash
# ---------------------------------------------------------------------------


def block_hash(text: str) -> str:
    """SHA-256 hash of the canonicalized block text.

    Computed after cleaning and before metadata enrichment so that cosmetic
    whitespace changes do not trigger unnecessary re-ingestion.

    Args:
        text: Cleaned block text (may contain newlines/extra spaces).

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    canonical = canonicalize(text)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Chunk signature
# ---------------------------------------------------------------------------


def chunk_signature(block_hashes: list[str]) -> str:
    """SHA-256 of a pipe-joined sequence of block hashes.

    Uniquely identifies a chunk's composition.  Changes when any constituent
    block changes, or when blocks are reordered.

    Args:
        block_hashes: Ordered list of ``block_hash()`` values for the
            blocks that make up this chunk.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    joined = "|".join(block_hashes)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Document fingerprint
# ---------------------------------------------------------------------------


def file_fingerprint(path: str | Path) -> str:
    """SHA-256 hash of the raw file bytes.

    Used to detect whether a file has changed between ingest runs without
    relying on filesystem mtime (which may be unreliable after copies or
    deployments).

    Args:
        path: Path to the file.

    Returns:
        64-character lowercase hex SHA-256 digest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint_bytes(raw_bytes: bytes) -> str:
    """SHA-256 hash of already-loaded raw bytes.

    Use this when the file has already been read into memory (e.g. inside
    the ingest pipeline after the Loader stage) to avoid a second disk read.

    Args:
        raw_bytes: Raw file content.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    return hashlib.sha256(raw_bytes).hexdigest()


def make_doc_id(source_path: str, content_hash: str) -> str:
    """Derive a stable doc_id from source path and content fingerprint.

    The doc_id changes when either the source path or the file content
    changes, which makes it suitable as a cache key in the DocStore.

    Args:
        source_path: Absolute path to the source file.
        content_hash: ``fingerprint_bytes()`` or ``file_fingerprint()`` value.

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    combined = f"{source_path}:{content_hash}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Block diff
# ---------------------------------------------------------------------------


@dataclass
class BlockDiffResult:
    """Result of comparing old vs new block hash sequences.

    Attributes:
        unchanged: Block hashes present in both old and new sequences.
        added: Block hashes present only in the new sequence.
        removed: Block hashes present only in the old sequence.
    """

    unchanged: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def unchanged_count(self) -> int:
        """Number of unchanged blocks."""
        return len(self.unchanged)

    @property
    def added_count(self) -> int:
        """Number of added blocks."""
        return len(self.added)

    @property
    def removed_count(self) -> int:
        """Number of removed blocks."""
        return len(self.removed)

    @property
    def total_new(self) -> int:
        """Total blocks in the new version."""
        return len(self.unchanged) + len(self.added)

    @property
    def has_changes(self) -> bool:
        """True if any blocks were added or removed."""
        return bool(self.added or self.removed)


def diff_blocks(
    old_hashes: list[str],
    new_hashes: list[str],
) -> BlockDiffResult:
    """Classify block hashes as unchanged, added, or removed.

    Comparison is set-based: a block is *unchanged* if its hash appears in
    both sequences, *added* if it is new-only, *removed* if it is old-only.
    Duplicate hashes within the same sequence are deduplicated before
    comparison so that re-ordered identical blocks do not inflate counts.

    Args:
        old_hashes: Ordered list of block hashes from the previous version.
        new_hashes: Ordered list of block hashes from the new version.

    Returns:
        ``BlockDiffResult`` with three classified lists.

    Examples:
        >>> r = diff_blocks(["a", "b", "c"], ["a", "b", "d"])
        >>> r.unchanged  # ["a", "b"]
        >>> r.added      # ["d"]
        >>> r.removed    # ["c"]
    """
    old_set = set(old_hashes)
    new_set = set(new_hashes)

    unchanged = [h for h in new_hashes if h in old_set]
    added = [h for h in new_hashes if h not in old_set]
    removed = [h for h in old_hashes if h not in new_set]

    return BlockDiffResult(unchanged=unchanged, added=added, removed=removed)
