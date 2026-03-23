"""Retrieval evaluation metrics — Recall@K, MRR, and nDCG@K.

All functions operate on ranked lists of retrieved chunk IDs compared
against a ground-truth set of relevant chunk IDs for a single query.

Conventions:
- ``retrieved``: ordered list of chunk IDs returned by the retrieval system,
  best first (index 0 is rank 1).
- ``relevant``: set (or list) of ground-truth relevant chunk IDs.
- ``k``: cut-off depth; only the top-k retrieved results are considered.

Each function returns a ``float`` in [0, 1] (or 0.0 when inputs are empty).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant items found in the top-k retrieved results.

    ``Recall@K = |relevant ∩ top_k(retrieved)| / |relevant|``

    Args:
        retrieved: Ordered list of retrieved chunk IDs (best first).
        relevant: Ground-truth relevant chunk IDs.
        k: Cut-off depth.

    Returns:
        Float in [0, 1]. Returns 0.0 when ``relevant`` is empty or k < 1.
    """
    if not relevant or k < 1:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for cid in retrieved[:k] if cid in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Reciprocal rank of the first relevant item in the top-k results.

    ``RR = 1 / rank_of_first_relevant``   (0.0 if none found in top-k)

    Args:
        retrieved: Ordered list of retrieved chunk IDs (best first).
        relevant: Ground-truth relevant chunk IDs.
        k: Cut-off depth.

    Returns:
        Float in [0, 1]. Returns 0.0 when no relevant item appears in top-k.
    """
    if not relevant or k < 1:
        return 0.0
    relevant_set = set(relevant)
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant_set:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Discounted Cumulative Gain at depth k (binary relevance).

    ``DCG@K = Σ_{i=1}^{K} rel_i / log2(i + 1)``

    where ``rel_i = 1`` if the item at rank i is relevant, else 0.

    Args:
        retrieved: Ordered list of retrieved chunk IDs (best first).
        relevant: Ground-truth relevant chunk IDs.
        k: Cut-off depth.

    Returns:
        Non-negative float (unbounded). Returns 0.0 for empty inputs.
    """
    if not relevant or k < 1:
        return 0.0
    relevant_set = set(relevant)
    score = 0.0
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant_set:
            score += 1.0 / math.log2(rank + 1)
    return score


def ideal_dcg_at_k(relevant: Sequence[str], k: int) -> float:
    """Ideal DCG at depth k (assumes all relevant items retrieved first).

    Args:
        relevant: Ground-truth relevant chunk IDs.
        k: Cut-off depth.

    Returns:
        Non-negative float representing the maximum achievable DCG@K.
    """
    if not relevant or k < 1:
        return 0.0
    n = min(len(relevant), k)
    return sum(1.0 / math.log2(rank + 1) for rank in range(1, n + 1))


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Normalised DCG at depth k (binary relevance).

    ``nDCG@K = DCG@K / IDCG@K``

    Returns 0.0 when IDCG is 0 (i.e., no relevant items exist).

    Args:
        retrieved: Ordered list of retrieved chunk IDs (best first).
        relevant: Ground-truth relevant chunk IDs.
        k: Cut-off depth.

    Returns:
        Float in [0, 1].
    """
    idcg = ideal_dcg_at_k(relevant, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


# ---------------------------------------------------------------------------
# Aggregate over a dataset
# ---------------------------------------------------------------------------


@dataclass
class QueryMetrics:
    """Per-query metric results.

    Attributes:
        query_id: Optional identifier for the query.
        recall_at_k: Recall at the configured cut-off.
        mrr: Reciprocal rank (same cut-off).
        ndcg_at_k: Normalised DCG at the configured cut-off.
        retrieved_count: Number of items returned by the retrieval system.
        relevant_count: Number of ground-truth relevant items.
        k: The cut-off depth used.
    """

    query_id: str = ""
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    retrieved_count: int = 0
    relevant_count: int = 0
    k: int = 10


@dataclass
class AggregateMetrics:
    """Macro-averaged metrics over an evaluation dataset.

    Attributes:
        mean_recall_at_k: Mean Recall@K across all queries.
        mrr: Mean Reciprocal Rank across all queries.
        mean_ndcg_at_k: Mean nDCG@K across all queries.
        num_queries: Number of queries evaluated.
        k: The cut-off depth used.
        per_query: Per-query metric breakdown (optional, may be empty).
    """

    mean_recall_at_k: float = 0.0
    mrr: float = 0.0
    mean_ndcg_at_k: float = 0.0
    num_queries: int = 0
    k: int = 10
    per_query: list[QueryMetrics] = field(default_factory=list)


def compute_aggregate_metrics(
    results: list[dict],
    k: int = 10,
) -> AggregateMetrics:
    """Compute macro-averaged Recall@K, MRR, and nDCG@K over a dataset.

    Each entry in ``results`` must contain:
    - ``"retrieved"``: ordered list of retrieved chunk IDs.
    - ``"relevant"``: list of ground-truth relevant chunk IDs.
    - ``"query_id"`` (optional): string identifier for the query.

    Args:
        results: List of per-query result dicts (see above).
        k: Cut-off depth. Defaults to 10.

    Returns:
        AggregateMetrics with macro averages and optional per-query breakdown.

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if not results:
        return AggregateMetrics(k=k)

    per_query: list[QueryMetrics] = []
    for entry in results:
        retrieved = entry.get("retrieved", [])
        relevant = entry.get("relevant", [])
        qid = entry.get("query_id", "")

        per_query.append(
            QueryMetrics(
                query_id=qid,
                recall_at_k=recall_at_k(retrieved, relevant, k),
                mrr=reciprocal_rank(retrieved, relevant, k),
                ndcg_at_k=ndcg_at_k(retrieved, relevant, k),
                retrieved_count=len(retrieved),
                relevant_count=len(relevant),
                k=k,
            )
        )

    n = len(per_query)
    return AggregateMetrics(
        mean_recall_at_k=sum(q.recall_at_k for q in per_query) / n,
        mrr=sum(q.mrr for q in per_query) / n,
        mean_ndcg_at_k=sum(q.ndcg_at_k for q in per_query) / n,
        num_queries=n,
        k=k,
        per_query=per_query,
    )
