"""Voyage AI cross-encoder reranker."""

import logging
import os

from rag.core.contracts.candidate import Candidate
from rag.core.interfaces.reranker import BaseReranker

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "rerank-2"
_DEFAULT_TOP_K = 8


class VoyageReranker(BaseReranker):
    """Cross-encoder reranker backed by the Voyage AI Rerank API.

    Sends candidate texts to ``voyageai.Client.rerank()`` and returns
    candidates ordered by descending ``relevance_score``. The score is
    stored on each ``Candidate`` in both ``rerank_score`` and
    ``final_score``.

    Usage::

        reranker = VoyageReranker(model="rerank-2", top_k=8)
        reranked = reranker.rerank(query, candidates, top_k=8)

    Args:
        model: Voyage rerank model identifier. Defaults to ``"rerank-2"``.
        top_k: Default maximum results to return. Can be overridden per call.
        api_key: Voyage AI API key. If None, reads from the
            ``VOYAGE_API_KEY`` environment variable.
        truncation: Whether to truncate documents that exceed the model's
            context window. Defaults to True.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        top_k: int = _DEFAULT_TOP_K,
        api_key: str | None = None,
        truncation: bool = True,
    ) -> None:
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "voyageai package is required for VoyageReranker. "
                "Install with: pip install voyageai"
            ) from exc

        self._model = model
        self._default_top_k = top_k
        self._truncation = truncation

        resolved_key = api_key or os.environ.get("VOYAGE_API_KEY")
        self._client = voyageai.Client(api_key=resolved_key)

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:
        """Rerank candidates using the Voyage AI cross-encoder.

        Sends ``stable_text`` from each candidate to the Voyage rerank API.
        Results are returned ordered by descending ``relevance_score``.

        Args:
            query: The user query string.
            candidates: Candidates to rerank.
            top_k: Maximum number of candidates to return after reranking.

        Returns:
            Reranked list of at most ``top_k`` Candidates with
            ``rerank_score`` and ``final_score`` populated.

        Raises:
            voyageai.error.VoyageError: On API-level errors.
        """
        if not candidates:
            return []

        effective_top_k = min(top_k, len(candidates))
        documents = [c.stable_text for c in candidates]

        try:
            response = self._client.rerank(
                query=query,
                documents=documents,
                model=self._model,
                top_k=effective_top_k,
                truncation=self._truncation,
            )
        except Exception as exc:
            logger.error("Voyage rerank API call failed: %s", exc)
            raise

        result: list[Candidate] = []
        for item in response.results:
            orig = candidates[item.index]
            score = float(item.relevance_score)
            result.append(
                orig.model_copy(
                    update={
                        "rerank_score": score,
                        "final_score": score,
                    }
                )
            )

        logger.debug(
            "VoyageReranker: reranked %d → %d candidates (model=%s, tokens=%d).",
            len(candidates),
            len(result),
            self._model,
            getattr(response, "total_tokens", 0),
        )
        return result
