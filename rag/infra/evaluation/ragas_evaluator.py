"""RAGAS-backed answer quality evaluator — Task 14.1.

Computes three RAGAS metrics for each (query, answer, contexts, ground_truth)
tuple:

- ``faithfulness``       — Is the answer grounded in the retrieved context?
- ``answer_relevancy``   — Does the answer address the question?
- ``context_precision``  — Are the top-retrieved passages relevant?

Install RAGAS::

    pip install ragas

When RAGAS is not installed the constructor raises a clear ``ImportError``
with installation instructions.  Callers in the eval pipeline should catch
this and display a friendly message in the UI.

Usage::

    evaluator = RagasEvaluator()
    scores = evaluator.evaluate(
        query="What is RAG?",
        answer="RAG stands for Retrieval-Augmented Generation.",
        contexts=["RAG combines retrieval with generation..."],
        ground_truth="RAG is a technique that retrieves relevant documents...",
    )
    # {"faithfulness": 0.95, "answer_relevancy": 0.88, "context_precision": 0.91}
"""

from __future__ import annotations

import logging
from typing import Any

from rag.core.interfaces.answer_evaluator import BaseAnswerEvaluator

logger = logging.getLogger(__name__)

_RAGAS_AVAILABLE = False
_ragas: Any = None

try:
    import ragas as _ragas  # type: ignore[import]
    _RAGAS_AVAILABLE = True
except ImportError:
    pass


class RagasEvaluator(BaseAnswerEvaluator):
    """Answer quality evaluator backed by the RAGAS library.

    Args:
        llm: Optional LLM instance to pass to RAGAS metrics.  If None,
            RAGAS will use its default LLM (requires OPENAI_API_KEY).
        embeddings: Optional embeddings instance for RAGAS.

    Raises:
        ImportError: At instantiation time if ``ragas`` is not installed.
    """

    def __init__(
        self,
        llm: Any = None,
        embeddings: Any = None,
    ) -> None:
        if not _RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS is not installed. Install it with:\n"
                "  pip install ragas\n"
                "For LangChain integration, also install:\n"
                "  pip install langchain langchain-openai"
            )
        self._llm = llm
        self._embeddings = embeddings

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = "",
    ) -> dict[str, float]:
        """Evaluate answer quality using RAGAS metrics.

        Args:
            query: The original user query.
            answer: The generated answer to evaluate.
            contexts: Retrieved context passages used to produce the answer.
            ground_truth: Optional reference answer for comparison.

        Returns:
            Dictionary with keys ``faithfulness``, ``answer_relevancy``,
            ``context_precision`` — all floats in [0.0, 1.0].

        Raises:
            RuntimeError: If RAGAS evaluation fails.
        """
        try:
            from datasets import Dataset  # type: ignore[import]
            from ragas import evaluate as ragas_evaluate  # type: ignore[import]
            from ragas.metrics import (  # type: ignore[import]
                answer_relevancy,
                context_precision,
                faithfulness,
            )

            data = {
                "question": [query],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth or ""],
            }
            dataset = Dataset.from_dict(data)

            kwargs: dict[str, Any] = {
                "dataset": dataset,
                "metrics": [faithfulness, answer_relevancy, context_precision],
            }
            if self._llm is not None:
                kwargs["llm"] = self._llm
            if self._embeddings is not None:
                kwargs["embeddings"] = self._embeddings

            result = ragas_evaluate(**kwargs)
            scores = result.to_pandas().iloc[0].to_dict()

            return {
                "faithfulness": float(scores.get("faithfulness", 0.0)),
                "answer_relevancy": float(scores.get("answer_relevancy", 0.0)),
                "context_precision": float(scores.get("context_precision", 0.0)),
            }

        except Exception as exc:
            raise RuntimeError(f"RAGAS evaluation failed: {exc}") from exc
