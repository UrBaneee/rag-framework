"""Windowed resync engine for incremental chunk updates — Task 11.3.

When a document is re-ingested after a small edit, most chunks are identical
to the previous version.  Re-embedding every chunk wastes tokens and latency.
``ResyncWindow`` limits re-embedding to chunks whose *block_hashes* differ
from the previous version, plus a configurable neighbourhood window around
each changed chunk.

Algorithm
---------
1. Build a mapping of ``chunk_signature → Chunk`` from the *old* chunks.
2. For each chunk in the *new* pack, check whether its ``chunk_signature``
   exists in the old mapping.
3. Mark new chunks as REUSE (old embedding can be copied) or REEMBED (must
   re-embed).
4. Expand the REEMBED set by ``window_size`` neighbours on each side so that
   context-sensitive re-ranking is not broken by stale adjacent embeddings.

This keeps embedding cost proportional to the *size of the edit*, not the
*size of the document*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.core.contracts.chunk import Chunk


@dataclass
class ResyncResult:
    """Output of a single ResyncWindow run.

    Attributes:
        reused: Chunks whose embeddings can be carried forward unchanged.
        reembed: Chunks that must be re-embedded (new or changed content).
        added_signatures: chunk_signatures that are brand-new (not in old set).
        removed_signatures: chunk_signatures from the old set not in the new pack.
        window_size: The neighbourhood window used during this run.
    """

    reused: list[Chunk] = field(default_factory=list)
    reembed: list[Chunk] = field(default_factory=list)
    added_signatures: list[str] = field(default_factory=list)
    removed_signatures: list[str] = field(default_factory=list)
    window_size: int = 1

    # ── Derived counts ──────────────────────────────────────────────────────

    @property
    def reused_count(self) -> int:
        """Number of chunks reused without re-embedding."""
        return len(self.reused)

    @property
    def reembed_count(self) -> int:
        """Number of chunks that will be re-embedded."""
        return len(self.reembed)

    @property
    def added_count(self) -> int:
        """Number of brand-new chunk signatures."""
        return len(self.added_signatures)

    @property
    def removed_count(self) -> int:
        """Number of chunk signatures from the old set no longer present."""
        return len(self.removed_signatures)

    @property
    def total_new(self) -> int:
        """Total chunks in the new pack."""
        return self.reused_count + self.reembed_count

    @property
    def savings_ratio(self) -> float:
        """Fraction of chunks that did NOT need re-embedding (0.0–1.0)."""
        if self.total_new == 0:
            return 1.0
        return self.reused_count / self.total_new


class ResyncWindow:
    """Windowed incremental resync engine.

    Compares a new chunk pack against the previous version and partitions
    chunks into *reuse* and *reembed* sets.  Only chunks within ``window_size``
    positions of a changed chunk are scheduled for re-embedding.

    Args:
        window_size: Number of neighbouring chunks on each side of a changed
            chunk that are also marked for re-embedding.  Defaults to 1.
            Set to 0 to re-embed only the directly changed chunks.

    Example::

        old_chunks = packer.pack(old_blocks)
        new_chunks = packer.pack(new_blocks)
        engine = ResyncWindow(window_size=1)
        result = engine.resync(old_chunks, new_chunks)
        # result.reused  — carry forward
        # result.reembed — call embedding provider on these
    """

    def __init__(self, window_size: int = 1) -> None:
        if window_size < 0:
            raise ValueError(f"window_size must be >= 0, got {window_size}")
        self.window_size = window_size

    def resync(
        self,
        old_chunks: list[Chunk],
        new_chunks: list[Chunk],
    ) -> ResyncResult:
        """Partition new_chunks into reuse and reembed sets.

        Args:
            old_chunks: Chunks from the previous ingest run (may have embeddings).
            new_chunks: Freshly packed chunks from the updated document.

        Returns:
            ``ResyncResult`` describing which chunks to reuse vs re-embed.
        """
        old_map: dict[str, Chunk] = {
            c.chunk_signature: c for c in old_chunks if c.chunk_signature
        }
        new_signatures: set[str] = {
            c.chunk_signature for c in new_chunks if c.chunk_signature
        }
        old_signatures: set[str] = set(old_map.keys())

        added_sigs = [
            c.chunk_signature
            for c in new_chunks
            if c.chunk_signature and c.chunk_signature not in old_signatures
        ]
        removed_sigs = [
            sig for sig in old_signatures if sig not in new_signatures
        ]

        # Identify positions of changed chunks in the new sequence
        changed_positions: set[int] = set()
        for i, chunk in enumerate(new_chunks):
            if not chunk.chunk_signature or chunk.chunk_signature not in old_map:
                changed_positions.add(i)

        # Expand by window_size on each side
        reembed_positions: set[int] = set()
        n = len(new_chunks)
        for pos in changed_positions:
            for offset in range(-self.window_size, self.window_size + 1):
                neighbour = pos + offset
                if 0 <= neighbour < n:
                    reembed_positions.add(neighbour)

        # Partition into reuse vs reembed
        reused: list[Chunk] = []
        reembed: list[Chunk] = []
        for i, chunk in enumerate(new_chunks):
            if i in reembed_positions:
                reembed.append(chunk)
            else:
                # Carry forward the old embedding if available
                old = old_map.get(chunk.chunk_signature or "")
                if old is not None and old.embedding is not None:
                    chunk = chunk.model_copy(update={"embedding": old.embedding})
                reused.append(chunk)

        return ResyncResult(
            reused=reused,
            reembed=reembed,
            added_signatures=added_sigs,
            removed_signatures=removed_sigs,
            window_size=self.window_size,
        )
