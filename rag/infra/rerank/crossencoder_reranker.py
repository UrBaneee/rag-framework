"""Local cross-encoder reranker using sentence-transformers.

Uses a cross-encoder model to score (query, passage) pairs with full
cross-attention, giving much better relevance judgements than the
bi-encoder embeddings used during retrieval.

Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
- ~80 MB, no API key required
- Trained on MS MARCO passage ranking (English)
- Fast inference on CPU

Usage::

    reranker = CrossEncoderReranker()
    candidates = reranker.rerank(query, fused_candidates, top_k=6)
"""

import logging
from typing import Optional

from rag.core.contracts.candidate import Candidate
from rag.core.interfaces.reranker import BaseReranker

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Reranker backed by a local sentence-transformers CrossEncoder.

    Scores every (query, passage) pair using the cross-encoder model and
    re-orders candidates by descending score.  Because the cross-encoder
    reads both inputs jointly with full attention, it is far more accurate
    at relevance judgement than the bi-encoder retrieval stage.

    Args:
        model: HuggingFace model name or local path.
            Defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2``.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``).
            If None the best available device is selected automatically.
        batch_size: Number of pairs to score per forward pass.

    Usage::

        reranker = CrossEncoderReranker()
        reranker = CrossEncoderReranker(model="cross-encoder/ms-marco-TinyBERT-L-2-v2")
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        load_kwargs: dict = {}
        if device is not None:
            load_kwargs["device"] = device

        self._model_name = model
        self._batch_size = batch_size
        self._encoder = CrossEncoder(model, **load_kwargs)

        logger.info(
            "CrossEncoderReranker loaded model '%s' on device '%s'.",
            model,
            device or "auto",
        )

    def rerank(
        self,
        query: str,
        candidates: list[Candidate],
        top_k: int,
    ) -> list[Candidate]:
        """Score all candidates against the query and return the top-k.

        Each candidate's ``stable_text`` (falling back to ``display_text``)
        is paired with the query and scored by the cross-encoder.  Candidates
        are sorted by descending score; the top-k are returned with
        ``rerank_score`` and ``final_score`` set.

        Args:
            query: The user query string.
            candidates: Candidates from the RRF fusion stage.
            top_k: Maximum number of candidates to return.

        Returns:
            Up to ``top_k`` candidates re-ordered by cross-encoder score,
            with ``rerank_score`` and ``final_score`` populated.
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [
            [query, cand.stable_text or cand.display_text or ""]
            for cand in candidates
        ]

        scores: list[float] = self._encoder.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        ).tolist()

        logger.debug(
            "CrossEncoderReranker: scored %d candidates; score range [%.3f, %.3f].",
            len(scores),
            min(scores),
            max(scores),
        )

        # Attach scores and sort descending
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        result = []
        for score, cand in scored[:top_k]:
            result.append(
                cand.model_copy(
                    update={
                        "rerank_score": float(score),
                        "final_score": float(score),
                    }
                )
            )

        return result
