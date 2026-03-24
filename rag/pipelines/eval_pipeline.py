"""Evaluation pipeline — Tasks 10.2, 10.3, and 14.2.

Computes retrieval metrics (Recall@K, MRR, nDCG@K), source attribution
diagnostics (bm25_only / vector_only / both ratios), and system
efficiency metrics (token savings, ingest/query latency) over a
labelled evaluation dataset.

Each entry in ``results`` may contain:

Required:
- ``"retrieved"``: ordered list of retrieved chunk IDs.
- ``"relevant"``: list of ground-truth relevant chunk IDs.

Optional:
- ``"query_id"``: string identifier for the query.
- ``"candidates"``: list of ``Candidate`` objects (or dicts with
  ``source_label``) for attribution diagnostics.
- ``"query_latency_ms"``: wall-clock time for this query in ms.
- ``"candidate_tokens"``: total token count across all retrieved candidates.
- ``"packed_tokens"``: tokens actually packed into the LLM context.

Usage::

    from rag.pipelines.eval_pipeline import run_eval

    results = [
        {
            "query_id": "q1",
            "retrieved": ["c1", "c2", "c3"],
            "relevant": ["c1", "c3"],
            "candidates": [...],
            "query_latency_ms": 42.0,
            "candidate_tokens": 800,
            "packed_tokens": 300,
        },
    ]
    report = run_eval(results, k=5, ingest_latency_ms=120.0)
    print(report.efficiency.token_saved_est)
    print(report.efficiency.mean_query_latency_ms)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from rag.core.contracts.eval_report import (
    AnswerQualityMetrics,
    EfficiencyMetrics,
    EvalReport,
    QueryEvalResult,
    SourceAttributionStats,
)
from rag.pipelines.scoring.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source attribution helpers
# ---------------------------------------------------------------------------


def _count_source_labels(candidates: Sequence[Any]) -> tuple[int, int, int]:
    """Count bm25_only / vector_only / both labels across candidates.

    Args:
        candidates: Iterable of objects with a ``source_label`` attribute
            or ``"source_label"`` dict key.

    Returns:
        Tuple of (bm25_only_count, vector_only_count, both_count).
    """
    bm25_only = 0
    vector_only = 0
    both = 0
    for cand in candidates:
        if hasattr(cand, "source_label"):
            label = cand.source_label
        elif isinstance(cand, dict):
            label = cand.get("source_label", "")
        else:
            label = ""

        if label == "bm25_only":
            bm25_only += 1
        elif label == "vector_only":
            vector_only += 1
        elif label == "both":
            both += 1

    return bm25_only, vector_only, both


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_eval(
    results: list[dict],
    k: int = 10,
    ingest_latency_ms: Optional[float] = None,
) -> EvalReport:
    """Run retrieval evaluation and return a full EvalReport.

    Computes macro-averaged Recall@K, MRR, and nDCG@K over ``results``,
    source attribution ratios from the ``"candidates"`` field, and
    system efficiency metrics from optional per-entry timing/token fields.

    Args:
        results: List of per-query result dicts.  Required keys:
            ``"retrieved"``, ``"relevant"``.  Optional keys:
            ``"query_id"``, ``"candidates"``, ``"query_latency_ms"``,
            ``"candidate_tokens"``, ``"packed_tokens"``.
        k: Cut-off depth.  Must be >= 1.
        ingest_latency_ms: Wall-clock time for the preceding ingest run
            (milliseconds).  Pass the value from ``IngestResult.elapsed_ms``
            or a similar source.  ``None`` if not available.

    Returns:
        EvalReport with aggregate metrics, source attribution, and
        efficiency diagnostics.

    Raises:
        ValueError: If k < 1.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    if not results:
        return EvalReport(k=k)

    per_query: list[QueryEvalResult] = []
    total_bm25_only = 0
    total_vector_only = 0
    total_both = 0
    total_candidates = 0

    # Efficiency accumulators
    query_latencies: list[float] = []
    token_savings: list[float] = []

    for entry in results:
        retrieved: list[str] = entry.get("retrieved", [])
        relevant: list[str] = entry.get("relevant", [])
        candidates: list[Any] = entry.get("candidates", [])
        qid: str = entry.get("query_id", "")
        expected_behavior: str = entry.get("expected_behavior", "")

        # Retrieval metrics
        r_at_k = recall_at_k(retrieved, relevant, k)
        rr = reciprocal_rank(retrieved, relevant, k)
        n_at_k = ndcg_at_k(retrieved, relevant, k)

        # Source attribution
        bm25_only, vector_only, both = _count_source_labels(candidates)
        total_bm25_only += bm25_only
        total_vector_only += vector_only
        total_both += both
        total_candidates += len(candidates)

        # Efficiency: query latency
        q_lat = entry.get("query_latency_ms")
        if q_lat is not None:
            query_latencies.append(float(q_lat))

        # Efficiency: token savings (candidate_tokens - packed_tokens)
        cand_tok = entry.get("candidate_tokens")
        pack_tok = entry.get("packed_tokens")
        if cand_tok is not None and pack_tok is not None:
            token_savings.append(float(cand_tok) - float(pack_tok))

        # Derive actual behavior from retrieval results:
        # "answer" if any candidates were retrieved, "abstain" otherwise.
        actual_behavior = "answer" if retrieved else "abstain"

        per_query.append(
            QueryEvalResult(
                query_id=qid,
                recall_at_k=r_at_k,
                mrr=rr,
                ndcg_at_k=n_at_k,
                retrieved_count=len(retrieved),
                relevant_count=len(relevant),
                k=k,
                bm25_only_count=bm25_only,
                vector_only_count=vector_only,
                both_count=both,
                expected_behavior=expected_behavior,
                actual_behavior=actual_behavior,
            )
        )

    n = len(per_query)
    mean_recall = sum(q.recall_at_k for q in per_query) / n
    mean_mrr = sum(q.mrr for q in per_query) / n
    mean_ndcg = sum(q.ndcg_at_k for q in per_query) / n

    # Source attribution ratios
    if total_candidates > 0:
        attr = SourceAttributionStats(
            bm25_only=total_bm25_only / total_candidates,
            vector_only=total_vector_only / total_candidates,
            both=total_both / total_candidates,
            total_candidates=total_candidates,
        )
    else:
        attr = SourceAttributionStats(total_candidates=0)

    # Efficiency metrics
    efficiency = EfficiencyMetrics(
        token_saved_est=(sum(token_savings) / len(token_savings)) if token_savings else None,
        mean_ingest_latency_ms=ingest_latency_ms,
        mean_query_latency_ms=(
            sum(query_latencies) / len(query_latencies) if query_latencies else None
        ),
        # skipped_chunks and changed_chunks are null until Task 11.2 (block-diff)
        skipped_chunks=None,
        changed_chunks=None,
    )

    return EvalReport(
        mean_recall_at_k=mean_recall,
        mrr=mean_mrr,
        mean_ndcg_at_k=mean_ndcg,
        num_queries=n,
        k=k,
        source_attribution=attr,
        efficiency=efficiency,
        per_query=per_query,
    )


def run_golden_eval(
    golden_entries: list[dict],
    query_pipeline: Any,
    evaluator: Any = None,
    trace_store: Any = None,
) -> AnswerQualityMetrics:
    """Run RAGAS answer quality evaluation over a golden test set.

    Each entry must contain ``"query"``, ``"expected_answer"``, and
    optionally ``"expected_sources"``.  The ``query_pipeline`` is called
    for each query to obtain an answer and context passages.

    Args:
        golden_entries: List of golden test set entries loaded from
            ``tests/fixtures/golden_answer_set.json``.
        query_pipeline: A ``QueryPipeline`` instance (or any object with
            a ``run(query)`` method returning a ``QueryResult``).
        evaluator: A ``BaseAnswerEvaluator`` instance.  If None or if
            RAGAS is not installed, returns ``AnswerQualityMetrics`` with
            ``ragas_available=False``.
        trace_store: Optional ``BaseTraceStore`` to record results.

    Returns:
        ``AnswerQualityMetrics`` with aggregate faithfulness,
        answer_relevancy, and context_precision scores.
    """
    if evaluator is None:
        logger.info("No answer evaluator provided — skipping RAGAS evaluation.")
        return AnswerQualityMetrics(ragas_available=False)

    per_query_scores: list[dict] = []
    faithfulness_vals: list[float] = []
    relevancy_vals: list[float] = []
    precision_vals: list[float] = []

    for entry in golden_entries:
        query = entry.get("query", "")
        ground_truth = entry.get("expected_answer", "")
        if not query:
            continue

        try:
            result = query_pipeline.run(query)
            answer_text = result.answer.text if result.answer else ""
            contexts = [c.display_text for c in result.candidates[:5]]

            scores = evaluator.evaluate(
                query=query,
                answer=answer_text,
                contexts=contexts,
                ground_truth=ground_truth,
            )

            per_query_scores.append({"query": query, **scores})
            faithfulness_vals.append(scores.get("faithfulness", 0.0))
            relevancy_vals.append(scores.get("answer_relevancy", 0.0))
            precision_vals.append(scores.get("context_precision", 0.0))

            if trace_store is not None:
                trace_store.save_run(
                    run_type="golden_eval_result",
                    metadata={"query": query, **scores},
                )

        except Exception as exc:
            logger.warning("RAGAS evaluation failed for query '%s': %s", query, exc)

    n = len(faithfulness_vals)
    if n == 0:
        return AnswerQualityMetrics(ragas_available=True, num_evaluated=0)

    return AnswerQualityMetrics(
        mean_faithfulness=sum(faithfulness_vals) / n,
        mean_answer_relevancy=sum(relevancy_vals) / n,
        mean_context_precision=sum(precision_vals) / n,
        num_evaluated=n,
        ragas_available=True,
        per_query=per_query_scores,
    )
