"""Query pipeline — retrieval, fusion, optional reranking, and answer generation.

This module is built incrementally across Phase 5 tasks:
  5.1  — attribute_candidates() source attribution merger
  5.2  — RRF fusion (rrf_fusion.py)
  5.3  — Full QueryPipeline (BM25 + vector → fusion → context → answer)
  5.4+ — Reranking, context packing, citation construction
"""

import logging
from typing import Optional

from rag.core.contracts.candidate import Candidate, RetrievalSource

logger = logging.getLogger(__name__)


def attribute_candidates(
    bm25_results: list[Candidate],
    vector_results: list[Candidate],
) -> list[Candidate]:
    """Merge BM25 and vector candidate lists with source attribution.

    Candidates appearing in only one list are marked ``bm25_only`` or
    ``vector_only``. Candidates appearing in both lists are merged into a
    single ``Candidate`` carrying both scores and marked ``both``.

    The merged list preserves all unique chunk_ids from both inputs.
    Order within each source list is not guaranteed to be preserved — callers
    should apply RRF fusion or another ranking step afterwards.

    Args:
        bm25_results: Candidates returned by the BM25 keyword index.
        vector_results: Candidates returned by the FAISS vector index.

    Returns:
        Merged list of Candidates with ``retrieval_source`` and score fields
        set to reflect which system(s) surfaced each candidate.
    """
    # Index BM25 results by chunk_id
    bm25_by_id: dict[str, Candidate] = {c.chunk_id: c for c in bm25_results}
    # Index vector results by chunk_id
    vec_by_id: dict[str, Candidate] = {c.chunk_id: c for c in vector_results}

    merged: dict[str, Candidate] = {}

    # BM25-only or both
    for chunk_id, bm25_cand in bm25_by_id.items():
        if chunk_id in vec_by_id:
            vec_cand = vec_by_id[chunk_id]
            merged[chunk_id] = bm25_cand.model_copy(
                update={
                    "vector_score": vec_cand.vector_score,
                    "retrieval_source": RetrievalSource.HYBRID,
                }
            )
        else:
            merged[chunk_id] = bm25_cand.model_copy(
                update={"retrieval_source": RetrievalSource.BM25}
            )

    # Vector-only (not already in merged)
    for chunk_id, vec_cand in vec_by_id.items():
        if chunk_id not in merged:
            merged[chunk_id] = vec_cand.model_copy(
                update={"retrieval_source": RetrievalSource.VECTOR}
            )

    result = list(merged.values())
    logger.debug(
        "attribute_candidates: %d bm25, %d vector → %d merged "
        "(%d bm25_only, %d vector_only, %d both)",
        len(bm25_results),
        len(vector_results),
        len(result),
        sum(1 for c in result if c.source_label == "bm25_only"),
        sum(1 for c in result if c.source_label == "vector_only"),
        sum(1 for c in result if c.source_label == "both"),
    )
    return result
