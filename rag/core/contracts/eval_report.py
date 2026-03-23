"""Evaluation report contracts — Tasks 10.2 and 10.3.

Defines data-classes for retrieval evaluation results, including
retrieval metrics, source attribution diagnostics, and system
efficiency metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceAttributionStats:
    """Counts and ratios of candidate source attribution labels.

    Each candidate is produced by BM25 only, vector search only,
    or both retrieval systems.  These ratios help diagnose whether
    the hybrid retrieval is balanced or skewed toward one system.

    Attributes:
        bm25_only: Fraction of candidates retrieved exclusively by BM25.
        vector_only: Fraction of candidates retrieved exclusively by vector.
        both: Fraction of candidates retrieved by both BM25 and vector.
        total_candidates: Raw count of candidates across all queries.
    """

    bm25_only: float = 0.0
    vector_only: float = 0.0
    both: float = 0.0
    total_candidates: int = 0

    def as_dict(self) -> dict:
        """Return a plain dict representation."""
        return {
            "bm25_only": self.bm25_only,
            "vector_only": self.vector_only,
            "both": self.both,
            "total_candidates": self.total_candidates,
        }


@dataclass
class EfficiencyMetrics:
    """System efficiency metrics for an evaluation run.

    Captures latency and token-savings estimates.  Fields that depend on
    block-diff results (``skipped_chunks``, ``changed_chunks``) are always
    ``None`` until Task 11.2 is complete.

    Attributes:
        token_saved_est: Mean tokens saved per query by context packing
            (candidate_tokens − packed_tokens).  ``None`` if not supplied.
        mean_ingest_latency_ms: Mean wall-clock time per ingest run (ms).
            ``None`` if not supplied.
        mean_query_latency_ms: Mean wall-clock time per query (ms).
            ``None`` if not supplied.
        skipped_chunks: Chunks skipped during incremental ingest (null until
            Task 11.2 block-diff is available).
        changed_chunks: Chunks re-ingested due to content change (null until
            Task 11.2 block-diff is available).
    """

    token_saved_est: Optional[float] = None
    mean_ingest_latency_ms: Optional[float] = None
    mean_query_latency_ms: Optional[float] = None
    skipped_chunks: Optional[int] = None
    changed_chunks: Optional[int] = None

    def as_dict(self) -> dict:
        """Return a plain dict representation."""
        return {
            "token_saved_est": self.token_saved_est,
            "mean_ingest_latency_ms": self.mean_ingest_latency_ms,
            "mean_query_latency_ms": self.mean_query_latency_ms,
            "skipped_chunks": self.skipped_chunks,
            "changed_chunks": self.changed_chunks,
        }


@dataclass
class QueryEvalResult:
    """Per-query evaluation outcome.

    Attributes:
        query_id: Optional identifier for the query.
        recall_at_k: Recall@K for this query.
        mrr: Reciprocal rank for this query.
        ndcg_at_k: nDCG@K for this query.
        retrieved_count: Number of candidates returned.
        relevant_count: Number of ground-truth relevant chunks.
        k: Cut-off depth used.
        bm25_only_count: Candidates attributed to BM25 only.
        vector_only_count: Candidates attributed to vector only.
        both_count: Candidates attributed to both systems.
    """

    query_id: str = ""
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    retrieved_count: int = 0
    relevant_count: int = 0
    k: int = 10
    bm25_only_count: int = 0
    vector_only_count: int = 0
    both_count: int = 0


@dataclass
class EvalReport:
    """Complete evaluation report with retrieval metrics and diagnostics.

    Aggregates macro-averaged retrieval metrics (Recall@K, MRR, nDCG@K)
    and source attribution diagnostics (bm25_only / vector_only / both
    ratios) across an evaluation dataset.

    Attributes:
        mean_recall_at_k: Macro-averaged Recall@K.
        mrr: Mean Reciprocal Rank.
        mean_ndcg_at_k: Macro-averaged nDCG@K.
        num_queries: Number of queries evaluated.
        k: Cut-off depth.
        source_attribution: Source attribution statistics across all candidates.
        efficiency: System efficiency metrics (latency, token savings).
        per_query: Per-query breakdown (optional).
    """

    mean_recall_at_k: float = 0.0
    mrr: float = 0.0
    mean_ndcg_at_k: float = 0.0
    num_queries: int = 0
    k: int = 10
    source_attribution: SourceAttributionStats = field(
        default_factory=SourceAttributionStats
    )
    efficiency: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    per_query: list[QueryEvalResult] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Return a plain dict representation."""
        return {
            "mean_recall_at_k": self.mean_recall_at_k,
            "mrr": self.mrr,
            "mean_ndcg_at_k": self.mean_ndcg_at_k,
            "num_queries": self.num_queries,
            "k": self.k,
            "source_attribution": self.source_attribution.as_dict(),
            "efficiency": self.efficiency.as_dict(),
        }
