"""Reciprocal Rank Fusion (RRF) — fuses multiple ranked candidate lists."""

import logging

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.interfaces.fusion import BaseFusion

logger = logging.getLogger(__name__)

_DEFAULT_K = 60


class RRFFusion(BaseFusion):
    """Reciprocal Rank Fusion over arbitrary candidate lists.

    RRF computes a fusion score for each unique chunk by summing its
    reciprocal rank contribution from every list it appears in::

        rrf_score(chunk) = Σ  1 / (k + rank_i)

    where ``rank_i`` is the 1-based position of the chunk in list *i* and
    ``k`` is a smoothing constant (default 60, from the original RRF paper).

    Chunks that appear in multiple lists receive higher combined scores.
    Ties are broken deterministically by ``chunk_id`` (lexicographic).

    The ``retrieval_source`` of the output candidates is set based on which
    lists contributed:
    - Appears in all lists → ``HYBRID``
    - Appears in one list  → preserves the source from that list's candidate

    Usage::

        fusion = RRFFusion(k=60)
        ranked = fusion.fuse([bm25_results, vector_results])

    Args:
        k: Smoothing constant. Higher values reduce the impact of rank
           differences. Defaults to 60.
    """

    def __init__(self, k: int = _DEFAULT_K) -> None:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")
        self._k = k

    def fuse(self, ranked_lists: list[list[Candidate]]) -> list[Candidate]:
        """Fuse ranked candidate lists using RRF scoring.

        Args:
            ranked_lists: Two or more ranked lists of Candidates.
                Each list must be ordered by descending relevance.

        Returns:
            Single list of Candidates ordered by descending ``rrf_score``.
            ``rrf_score`` and ``final_score`` are populated on every result.
            Returns an empty list if all input lists are empty.

        Raises:
            ValueError: If ``ranked_lists`` is empty.
        """
        if not ranked_lists:
            raise ValueError("ranked_lists must contain at least one list.")

        # Accumulate RRF scores and track best candidate metadata per chunk
        rrf_scores: dict[str, float] = {}
        # Best candidate object seen for each chunk_id (from any list)
        best_candidate: dict[str, Candidate] = {}
        # Track which list indices contributed to each chunk_id
        contributing_lists: dict[str, set[int]] = {}

        for list_idx, candidates in enumerate(ranked_lists):
            for rank_0, candidate in enumerate(candidates):
                chunk_id = candidate.chunk_id
                rank_1 = rank_0 + 1  # 1-based rank
                score_contribution = 1.0 / (self._k + rank_1)

                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score_contribution

                if chunk_id not in best_candidate:
                    best_candidate[chunk_id] = candidate
                    contributing_lists[chunk_id] = {list_idx}
                else:
                    contributing_lists[chunk_id].add(list_idx)
                    # Merge scores from different retrieval systems
                    existing = best_candidate[chunk_id]
                    updates: dict = {}
                    if candidate.bm25_score is not None and existing.bm25_score is None:
                        updates["bm25_score"] = candidate.bm25_score
                    if candidate.vector_score is not None and existing.vector_score is None:
                        updates["vector_score"] = candidate.vector_score
                    if updates:
                        best_candidate[chunk_id] = existing.model_copy(update=updates)

        # Build output candidates with rrf_score / final_score / source set
        result: list[Candidate] = []
        for chunk_id, rrf_score in rrf_scores.items():
            base = best_candidate[chunk_id]
            n_contributing = len(contributing_lists[chunk_id])

            # Determine source: all lists contributed → HYBRID; else preserve
            if n_contributing >= len(ranked_lists) and len(ranked_lists) > 1:
                source = RetrievalSource.HYBRID
            else:
                source = base.retrieval_source

            result.append(
                base.model_copy(
                    update={
                        "rrf_score": rrf_score,
                        "final_score": rrf_score,
                        "retrieval_source": source,
                    }
                )
            )

        # Sort by descending rrf_score; break ties deterministically by chunk_id
        result.sort(key=lambda c: (-c.rrf_score, c.chunk_id))

        logger.debug(
            "RRF fused %d lists → %d unique candidates (k=%d).",
            len(ranked_lists),
            len(result),
            self._k,
        )
        return result
