"""Abstract base class for answer quality evaluator plugins — Task 14.1."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAnswerEvaluator(ABC):
    """Interface for answer quality evaluation plugins.

    Implementations measure how well a generated answer aligns with the
    retrieved context (faithfulness), addresses the query (answer_relevancy),
    and whether the retrieved contexts were useful (context_precision).
    """

    @abstractmethod
    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> dict[str, float]:
        """Evaluate the quality of a generated answer.

        Args:
            query: The original user query.
            answer: The generated answer to evaluate.
            contexts: List of context passage strings used to generate the answer.
            ground_truth: Optional reference answer for comparison.

        Returns:
            Dictionary mapping metric names to float scores in [0.0, 1.0].
            Required keys: ``faithfulness``, ``answer_relevancy``,
            ``context_precision``.

        Raises:
            ImportError: If the underlying evaluation library is not installed.
            RuntimeError: If evaluation fails.
        """
