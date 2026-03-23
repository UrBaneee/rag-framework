"""No-op reranker — passes candidates through unchanged as a fallback."""

import logging

from rag.core.contracts.candidate import Candidate
from rag.core.interfaces.reranker import BaseReranker

logger = logging.getLogger(__name__)


class NoOpReranker(BaseReranker):
    """Reranker that returns candidates in their original order unchanged.

    Used as a fallback when no external reranker is configured. Preserves
    the RRF-fused ranking and sets ``rerank_score`` to ``rrf_score`` so that
    ``final_score`` is always populated for downstream consumers.

    Usage::

        reranker = NoOpReranker()
        candidates = reranker.rerank(query, fused_candidates, top_k=5)
    """

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:
        """Return the top-k candidates unchanged, with rerank_score = rrf_score.

        Args:
            query: The user query string (unused by this implementation).
            candidates: Candidates from the fusion stage.
            top_k: Maximum number of candidates to return.

        Returns:
            First ``top_k`` candidates with ``rerank_score`` and
            ``final_score`` set to ``rrf_score``.
        """
        result = []
        for cand in candidates[:top_k]:
            result.append(
                cand.model_copy(
                    update={
                        "rerank_score": cand.rrf_score,
                        "final_score": cand.rrf_score,
                    }
                )
            )
        logger.debug(
            "NoOpReranker: passed %d/%d candidates through unchanged.",
            len(result),
            len(candidates),
        )
        return result
