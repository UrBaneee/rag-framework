"""Hybrid resync engine with tracing support — Task 11.3.

``ResyncHybrid`` wraps ``ResyncWindow`` and emits structured ``ResyncStats``
after each run so that the ingest pipeline can record efficiency metrics
and surface them in the Evaluation panel (Task 10.3 / 10.6).

The "hybrid" name reflects that it combines:
- Windowed chunk reuse  (from ``ResyncWindow``)
- Full-document fallback (when too many chunks have changed)
- Traceable stats output (for observability)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag.core.contracts.chunk import Chunk
from rag.infra.chunking.resync_window import ResyncResult, ResyncWindow

logger = logging.getLogger(__name__)


@dataclass
class ResyncStats:
    """Summary statistics emitted by ``ResyncHybrid`` after a resync run.

    Attributes:
        reused_count: Chunks carried forward without re-embedding.
        reembed_count: Chunks scheduled for re-embedding.
        added_count: Brand-new chunk signatures.
        removed_count: Old chunk signatures no longer present.
        savings_ratio: Fraction of chunks NOT re-embedded (0.0–1.0).
        fallback_used: True if the full-document fallback was triggered.
        window_size: Neighbourhood window applied during this run.
    """

    reused_count: int
    reembed_count: int
    added_count: int
    removed_count: int
    savings_ratio: float
    fallback_used: bool
    window_size: int

    @property
    def skipped_chunks(self) -> int:
        """Alias for ``reused_count`` — used by efficiency metrics."""
        return self.reused_count

    @property
    def changed_chunks(self) -> int:
        """Alias for ``reembed_count`` — used by efficiency metrics."""
        return self.reembed_count


class ResyncHybrid:
    """Windowed resync with full-document fallback and tracing.

    If the fraction of changed chunks exceeds ``fallback_threshold``, the
    engine falls back to scheduling all chunks for re-embedding (cheaper
    than a partial re-embed when almost everything changed anyway).

    Args:
        window_size: Neighbourhood window passed to ``ResyncWindow``.
            Defaults to 1.
        fallback_threshold: If the fraction of chunks that need re-embedding
            is >= this value, fall back to full re-embed.  Set to 1.0 to
            disable fallback.  Defaults to 0.8.

    Example::

        engine = ResyncHybrid(window_size=1)
        result, stats = engine.resync(old_chunks, new_chunks)
        # result.reembed — call embedding provider on these
        # stats.savings_ratio — fraction saved
    """

    def __init__(
        self,
        window_size: int = 1,
        fallback_threshold: float = 0.8,
    ) -> None:
        if not 0.0 <= fallback_threshold <= 1.0:
            raise ValueError(
                f"fallback_threshold must be in [0, 1], got {fallback_threshold}"
            )
        self._window = ResyncWindow(window_size=window_size)
        self.window_size = window_size
        self.fallback_threshold = fallback_threshold

    def resync(
        self,
        old_chunks: list[Chunk],
        new_chunks: list[Chunk],
    ) -> tuple[ResyncResult, ResyncStats]:
        """Run windowed resync and emit tracing stats.

        Args:
            old_chunks: Previous version's chunks (may carry embeddings).
            new_chunks: Freshly packed chunks from the updated document.

        Returns:
            A ``(ResyncResult, ResyncStats)`` tuple.  The result's
            ``reembed`` list is what should be passed to the embedding
            provider; ``reused`` can be stored directly.
        """
        result = self._window.resync(old_chunks, new_chunks)
        fallback_used = False

        # Full-document fallback: if too many chunks changed, re-embed all
        if (
            result.total_new > 0
            and (result.reembed_count / result.total_new) >= self.fallback_threshold
        ):
            logger.info(
                "ResyncHybrid: changed ratio %.2f >= threshold %.2f — "
                "falling back to full re-embed (%d chunks)",
                result.reembed_count / result.total_new,
                self.fallback_threshold,
                result.total_new,
            )
            result = ResyncResult(
                reused=[],
                reembed=list(new_chunks),
                added_signatures=result.added_signatures,
                removed_signatures=result.removed_signatures,
                window_size=self.window_size,
            )
            fallback_used = True

        stats = ResyncStats(
            reused_count=result.reused_count,
            reembed_count=result.reembed_count,
            added_count=result.added_count,
            removed_count=result.removed_count,
            savings_ratio=result.savings_ratio,
            fallback_used=fallback_used,
            window_size=self.window_size,
        )

        logger.debug(
            "ResyncHybrid stats: reused=%d reembed=%d added=%d removed=%d "
            "savings=%.1f%% fallback=%s",
            stats.reused_count,
            stats.reembed_count,
            stats.added_count,
            stats.removed_count,
            stats.savings_ratio * 100,
            fallback_used,
        )

        return result, stats
