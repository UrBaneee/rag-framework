"""Abstract base class for candidate list fusion plugins."""

from abc import ABC, abstractmethod

from rag.core.contracts.candidate import Candidate


class BaseFusion(ABC):
    """Interface that all candidate fusion implementations must implement.

    Fusion combines ranked candidate lists from different retrieval systems
    (BM25, vector, etc.) into a single unified ranking. Implementations
    include Reciprocal Rank Fusion (RRF) and score normalisation fusion.
    """

    @abstractmethod
    def fuse(self, ranked_lists: list[list[Candidate]]) -> list[Candidate]:
        """Fuse multiple ranked candidate lists into one ranked list.

        Args:
            ranked_lists: Two or more lists of Candidates, each ordered by
                descending relevance from a single retrieval system.
                Candidates may appear in more than one list (same chunk_id).

        Returns:
            A single list of Candidates ordered by descending fusion score,
            with ``rrf_score`` (or equivalent) and ``final_score`` populated.
        """
