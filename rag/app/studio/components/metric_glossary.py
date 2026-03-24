"""Metric glossary — centralized tooltip definitions for evaluation metrics.

All metric descriptions live here so that pages and components never
drift out of sync.  Import ``GLOSSARY`` and look up by metric key.

Usage::

    from rag.app.studio.components.metric_glossary import GLOSSARY, tooltip

    tip = tooltip("recall_at_k")   # returns the help string for st.metric
"""

from __future__ import annotations

from typing import TypedDict


class MetricEntry(TypedDict):
    """Schema for a single glossary entry."""

    label: str          # Human-readable display name
    description: str    # What the metric measures
    direction: str      # "higher_is_better" | "lower_is_better" | "neutral"
    pitfalls: str       # Common misinterpretation warnings


# ---------------------------------------------------------------------------
# Central glossary
# ---------------------------------------------------------------------------

GLOSSARY: dict[str, MetricEntry] = {
    # ------------------------------------------------------------------
    # Retrieval metrics
    # ------------------------------------------------------------------
    "recall_at_k": MetricEntry(
        label="Recall@K",
        description=(
            "Fraction of ground-truth relevant chunks that appear in the "
            "top-K retrieved results.  A score of 1.0 means every relevant "
            "chunk was found within the top-K."
        ),
        direction="higher_is_better",
        pitfalls=(
            "Recall ignores rank order — a system that returns all relevant "
            "chunks at rank K scores the same as one that returns them at "
            "rank 1.  Pair with MRR or nDCG to assess ranking quality."
        ),
    ),
    "mrr": MetricEntry(
        label="MRR (Mean Reciprocal Rank)",
        description=(
            "Average of 1/rank for the first relevant result across queries. "
            "MRR = 1.0 means the top result is always relevant; "
            "MRR = 0.5 means the first relevant result is typically at rank 2."
        ),
        direction="higher_is_better",
        pitfalls=(
            "MRR only cares about the single highest-ranked relevant item. "
            "It ignores all other relevant chunks, so it may overestimate "
            "quality when only the first result matters to the use case."
        ),
    ),
    "ndcg_at_k": MetricEntry(
        label="nDCG@K",
        description=(
            "Normalised Discounted Cumulative Gain at depth K.  Rewards "
            "retrieving relevant chunks early (higher ranks) and penalises "
            "relevant chunks pushed to lower ranks.  Score of 1.0 is ideal."
        ),
        direction="higher_is_better",
        pitfalls=(
            "nDCG assumes binary relevance (relevant / not relevant).  "
            "It does not capture partial relevance or answer quality.  "
            "A high nDCG@K does not guarantee a good generated answer."
        ),
    ),
    # ------------------------------------------------------------------
    # Source attribution
    # ------------------------------------------------------------------
    "bm25_only": MetricEntry(
        label="BM25-only ratio",
        description=(
            "Fraction of retrieved candidates that were found exclusively "
            "by BM25 keyword search and not by vector search."
        ),
        direction="neutral",
        pitfalls=(
            "A very high BM25-only ratio may indicate that vector embeddings "
            "are not indexed or are not well-calibrated for this corpus.  "
            "Aim for a balanced hybrid split."
        ),
    ),
    "vector_only": MetricEntry(
        label="Vector-only ratio",
        description=(
            "Fraction of retrieved candidates found exclusively by vector "
            "similarity search and not by BM25 keyword search."
        ),
        direction="neutral",
        pitfalls=(
            "A very high vector-only ratio may indicate BM25 is under-tuned "
            "or that the corpus is too small for keyword matching.  "
            "Check that BM25 is properly indexed."
        ),
    ),
    "both": MetricEntry(
        label="Hybrid (both) ratio",
        description=(
            "Fraction of candidates retrieved by both BM25 and vector search. "
            "High overlap means both systems agree on relevance — a good sign "
            "for retrieval robustness."
        ),
        direction="higher_is_better",
        pitfalls=(
            "Very high overlap can also mean both systems are redundant and "
            "a single retriever may suffice, reducing complexity."
        ),
    ),
    # ------------------------------------------------------------------
    # Efficiency metrics
    # ------------------------------------------------------------------
    "token_saved_est": MetricEntry(
        label="Token saved (est.)",
        description=(
            "Estimated mean tokens saved per query by context packing — "
            "the difference between total candidate tokens and the tokens "
            "actually packed into the LLM context window."
        ),
        direction="higher_is_better",
        pitfalls=(
            "This is an estimate based on character length divided by 4. "
            "Actual token counts vary by tokenizer and model.  "
            "Savings do not directly measure answer quality."
        ),
    ),
    "mean_query_latency_ms": MetricEntry(
        label="Mean query latency (ms)",
        description=(
            "Average wall-clock time in milliseconds to process a single "
            "query end-to-end through retrieval (and optionally generation)."
        ),
        direction="lower_is_better",
        pitfalls=(
            "Latency is heavily influenced by index size, hardware, and "
            "whether generation is enabled.  Compare only within the same "
            "configuration."
        ),
    ),
    "mean_ingest_latency_ms": MetricEntry(
        label="Ingest latency (ms)",
        description=(
            "Wall-clock time in milliseconds for the most recent ingest run. "
            "Includes parsing, chunking, embedding, and indexing."
        ),
        direction="lower_is_better",
        pitfalls=(
            "Ingest latency scales with document size and embedding batch size. "
            "A single large document can skew this metric significantly."
        ),
    ),
    "skipped_chunks": MetricEntry(
        label="Skipped chunks",
        description=(
            "Number of chunks skipped during incremental ingest because their "
            "content had not changed since the previous ingest run."
        ),
        direction="higher_is_better",
        pitfalls=(
            "Available only after Task 11.2 (block-diff) is complete. "
            "Shown as N/A until then."
        ),
    ),
    "changed_chunks": MetricEntry(
        label="Changed chunks",
        description=(
            "Number of chunks re-ingested because their content changed since "
            "the previous ingest run, as detected by block-level diff."
        ),
        direction="neutral",
        pitfalls=(
            "Available only after Task 11.2 (block-diff) is complete. "
            "Shown as N/A until then."
        ),
    ),
    # ------------------------------------------------------------------
    # Answer Quality (RAGAS)
    # ------------------------------------------------------------------
    "faithfulness": MetricEntry(
        label="Faithfulness",
        description=(
            "Measures whether the generated answer is grounded in the "
            "retrieved context passages.  A high score means every claim "
            "in the answer can be traced back to the provided context."
        ),
        direction="higher_is_better",
        pitfalls=(
            "Requires RAGAS and an LLM to compute.  Faithfulness can be "
            "high even if the answer is incomplete — it only checks "
            "that stated facts are supported, not that all relevant facts "
            "are included."
        ),
    ),
    "answer_relevancy": MetricEntry(
        label="Answer Relevancy",
        description=(
            "Measures how directly the generated answer addresses the "
            "original question.  Computed by asking the LLM to generate "
            "candidate questions from the answer and comparing them to "
            "the original query via embedding similarity."
        ),
        direction="higher_is_better",
        pitfalls=(
            "Requires RAGAS and an embedding model.  A concise but "
            "incomplete answer can still score high if the portion it "
            "covers is on-topic.  Does not penalise omissions."
        ),
    ),
    "context_precision": MetricEntry(
        label="Context Precision",
        description=(
            "Measures the fraction of retrieved context passages that are "
            "actually relevant to the question.  Higher precision means "
            "less noise in the context window, reducing hallucination risk."
        ),
        direction="higher_is_better",
        pitfalls=(
            "Requires RAGAS and a ground-truth answer or source list.  "
            "Low context precision may indicate that the retrieval step "
            "is pulling in off-topic chunks rather than a generation "
            "problem."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def tooltip(key: str) -> str:
    """Return a formatted tooltip string for use as st.metric(help=...).

    Args:
        key: Glossary key (e.g. ``"recall_at_k"``).

    Returns:
        Multi-line string with description, direction, and pitfalls.
        Falls back to the key name if not found.
    """
    entry = GLOSSARY.get(key)
    if entry is None:
        return key

    direction_icon = {
        "higher_is_better": "↑ Higher is better",
        "lower_is_better": "↓ Lower is better",
        "neutral": "→ Context-dependent",
    }.get(entry["direction"], "")

    return (
        f"{entry['description']}\n\n"
        f"**{direction_icon}**\n\n"
        f"⚠️ *{entry['pitfalls']}*"
    )


def label(key: str) -> str:
    """Return the human-readable label for a metric key.

    Args:
        key: Glossary key.

    Returns:
        Display label string, or the key itself if not found.
    """
    entry = GLOSSARY.get(key)
    return entry["label"] if entry else key
