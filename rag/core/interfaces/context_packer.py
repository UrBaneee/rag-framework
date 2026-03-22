"""Abstract base class for context packer plugins."""

from abc import ABC, abstractmethod

from rag.core.contracts.candidate import Candidate


class BaseContextPacker(ABC):
    """Interface for selecting and ordering candidates within a token budget.

    The ContextPacker takes the reranked candidate list and selects the
    top candidates that fit within a token budget for inclusion in the
    LLM prompt context. It is positioned between the reranker and the
    prompt builder in the query pipeline.

    Default configuration (from Section 10):
        context_top_k: 3
        token_budget: 3000
    """

    @abstractmethod
    def pack(self, candidates: list[Candidate], token_budget: int) -> list[Candidate]:
        """Select candidates that fit within the token budget.

        Args:
            candidates: Reranked candidates ordered by ``final_score``
                descending.
            token_budget: Maximum total tokens allowed across all selected
                candidates' ``display_text`` fields.

        Returns:
            Subset of candidates that fit within the budget, preserving
            rank order. May be shorter than the input list.
        """
