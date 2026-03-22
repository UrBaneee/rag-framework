"""Abstract base class for answer composer plugins."""

from abc import ABC, abstractmethod

from rag.core.contracts.answer import Answer
from rag.core.contracts.candidate import Candidate


class BaseAnswerComposer(ABC):
    """Interface for the answer composition stage of the query pipeline.

    The AnswerComposer orchestrates prompt building, LLM generation, and
    citation assembly. It receives packed context candidates and produces
    a grounded Answer with inline citations.

    Key generation rules (from Section 11):
    - Answers must rely only on retrieved evidence
    - The system abstains when evidence is insufficient
    - Citations are mandatory for all factual claims
    """

    @abstractmethod
    def compose(self, query: str, candidates: list[Candidate]) -> Answer:
        """Generate a grounded answer from a query and context candidates.

        Args:
            query: The original user query string.
            candidates: Context-packed candidates to use as evidence.
                These are the candidates selected by the ContextPacker.

        Returns:
            A grounded Answer with inline citations. If evidence is
            insufficient, returns an Answer with ``abstained=True``.
        """
