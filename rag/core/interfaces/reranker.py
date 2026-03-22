"""Abstract base class for reranker plugins."""

from abc import ABC, abstractmethod

from rag.core.contracts.candidate import Candidate


class BaseReranker(ABC):
    """Interface that all reranker plugins must implement.

    Rerankers score a pool of candidates against a query and return a
    re-ordered, possibly truncated list. Implementations include
    Voyage AI cross-encoder, Cohere reranker, and local cross-encoders.

    Reranking is applied after RRF fusion and before context packing.
    """

    @abstractmethod
    def rerank(self, query: str, candidates: list[Candidate], top_k: int) -> list[Candidate]:
        """Rerank a list of candidates with respect to the query.

        Args:
            query: The user query string.
            candidates: Candidates from the fusion stage, ordered by
                ``rrf_score`` descending.
            top_k: Maximum number of candidates to return after reranking.

        Returns:
            Reranked list of at most ``top_k`` Candidates, ordered by
            descending ``rerank_score``. The ``final_score`` field is set
            to ``rerank_score`` for all returned candidates.
        """
