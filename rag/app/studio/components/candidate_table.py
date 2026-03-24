"""Candidate table component — renders retrieval results in Streamlit."""

from __future__ import annotations

from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Score badge helpers
# ---------------------------------------------------------------------------

def _score_bar(score: float | None, max_val: float = 1.0) -> str:
    """Return a simple text progress bar for a score in [0, max_val]."""
    if score is None:
        return "—"
    frac = min(1.0, max(0.0, score / max_val)) if max_val > 0 else 0.0
    filled = int(frac * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score:.4f}"


def _source_badge(source: str) -> str:
    icons = {"bm25": "🔑", "vector": "🔢", "hybrid": "🔀"}
    return icons.get(source.lower(), "❓") + " " + source


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------


def render_candidate_table(
    candidates: list[dict[str, Any]],
    title: str = "Retrieved Chunks",
    pre_rerank_ids: list[str] | None = None,
) -> None:
    """Render a table of retrieved candidates with scores and source attribution.

    Args:
        candidates: List of candidate dicts with keys: chunk_id, doc_id,
            display_text, bm25_score, vector_score, rrf_score, rerank_score,
            final_score, retrieval_source, metadata.
        title: Section heading string.
        pre_rerank_ids: If provided, show a ↕ rerank-change indicator
            by comparing current order to pre-rerank order.
    """
    st.subheader(title)
    if not candidates:
        st.info("No candidates retrieved.")
        return

    pre_rank: dict[str, int] = {}
    if pre_rerank_ids:
        pre_rank = {cid: i for i, cid in enumerate(pre_rerank_ids)}

    for rank, cand in enumerate(candidates, start=1):
        chunk_id = cand.get("chunk_id", "")
        source = cand.get("retrieval_source", "")
        doc_id = cand.get("doc_id", "")
        display_text = cand.get("display_text", "")
        meta = cand.get("metadata", {})
        page = meta.get("start_page") or meta.get("page", "")

        # Rerank change indicator
        rerank_delta = ""
        if pre_rerank_ids and chunk_id in pre_rank:
            old_rank = pre_rank[chunk_id] + 1
            diff = old_rank - rank
            if diff > 0:
                rerank_delta = f" ▲{diff}"
            elif diff < 0:
                rerank_delta = f" ▼{abs(diff)}"
            else:
                rerank_delta = " ↔"

        label = (
            f"**#{rank}{rerank_delta}** · {_source_badge(source)} · "
            f"`{chunk_id[:12]}…` · {doc_id}"
            + (f" p.{page}" if page else "")
        )
        with st.expander(label, expanded=(rank <= 3)):
            # Score row
            # vector_score is stored as -L2_distance (higher = more similar).
            # Display the absolute value so readers see a plain distance where
            # lower = closer, and label it "L2 dist ↓" to make the direction clear.
            score_col1, score_col2, score_col3, score_col4 = st.columns(4)
            score_col1.metric("BM25 ↑", f"{cand.get('bm25_score'):.4f}" if cand.get('bm25_score') is not None else "—")
            raw_vec = cand.get('vector_score')
            score_col2.metric(
                "L2 dist ↓",
                f"{abs(raw_vec):.4f}" if raw_vec is not None else "—",
                help="L2 (Euclidean) distance from the query vector. Lower = more similar.",
            )
            score_col3.metric("RRF ↑", f"{cand.get('rrf_score', 0):.4f}")
            score_col4.metric(
                "Rerank ↑" if cand.get('rerank_score') is not None else "Final ↑",
                f"{(cand.get('rerank_score') or cand.get('final_score', 0)):.4f}",
            )
            # Text content
            st.markdown("**Chunk text:**")
            st.markdown(f"> {display_text[:500]}" + ("…" if len(display_text) > 500 else ""))
            if meta:
                st.json(meta, expanded=False)


def render_context_packing_details(
    packed_candidates: list[dict[str, Any]],
    all_candidates: list[dict[str, Any]],
    context_top_k: int,
    token_budget: int,
    packed_tokens: int,
    truncated: bool,
) -> None:
    """Render context packing details section.

    Shows selected chunks, dropped chunks with drop reason, and a packing
    summary (top_k, token_budget, packed tokens, truncated flag).

    Args:
        packed_candidates: The candidates that were packed into context.
        all_candidates: All candidates after reranking (superset of packed).
        context_top_k: The configured top-k for the packer.
        token_budget: The configured token budget.
        packed_tokens: Actual token count of packed context.
        truncated: Whether packing stopped early due to token budget.
    """
    st.subheader("📦 Context Packing")

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("context_top_k", context_top_k)
    c2.metric("token_budget", token_budget)
    c3.metric("Packed tokens", packed_tokens)
    c4.metric("Truncated", "Yes ⚠️" if truncated else "No ✅")

    packed_ids = {c.get("chunk_id") for c in packed_candidates}

    # Selected chunks
    with st.expander(f"✅ Selected chunks ({len(packed_candidates)})", expanded=True):
        if not packed_candidates:
            st.info("No chunks were packed.")
        for i, cand in enumerate(packed_candidates, start=1):
            st.markdown(
                f"**[{i}]** `{cand.get('chunk_id', '')[:16]}…` — "
                f"{cand.get('doc_id', '')} "
                f"(~{cand.get('metadata', {}).get('token_count', '?')} tokens)"
            )
            st.caption(cand.get("display_text", "")[:200])

    # Dropped chunks
    dropped = [c for c in all_candidates if c.get("chunk_id") not in packed_ids]
    if dropped:
        with st.expander(f"🚫 Dropped chunks ({len(dropped)})", expanded=False):
            seen_texts: set[str] = {c.get("stable_text", "").strip() for c in packed_candidates}
            for cand in dropped:
                stable = cand.get("stable_text", "").strip()
                if stable in seen_texts:
                    reason = "Duplicate (stable_text match)"
                elif truncated:
                    reason = "Token budget exceeded"
                else:
                    reason = "Below top-k limit"
                st.markdown(
                    f"- `{cand.get('chunk_id', '')[:16]}…` — **{reason}**"
                )


def render_answer_section(
    answer_text: str,
    citations: list[dict[str, Any]],
    abstained: bool,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    generation_latency_ms: float,
) -> None:
    """Render the final answer, citations, and generation usage stats.

    Args:
        answer_text: The LLM-generated answer text.
        citations: List of citation dicts (ref_number, chunk_id, source_label, …).
        abstained: Whether the model abstained from answering.
        prompt_tokens: Token count for the prompt.
        completion_tokens: Token count for the completion.
        total_tokens: Total token count.
        generation_latency_ms: Generation latency in milliseconds.
    """
    st.subheader("💬 Answer")

    if abstained:
        st.warning("⚠️ The model abstained — insufficient evidence in retrieved context.")

    st.markdown(answer_text if answer_text else "*(no answer generated)*")

    # Citations
    if citations:
        st.markdown("**Sources:**")
        for cit in citations:
            st.markdown(
                f"  [{cit.get('ref_number', '?')}] {cit.get('source_label', '')} "
                f"— `{cit.get('chunk_id', '')[:12]}…`"
            )

    # Generation usage stats
    st.subheader("📊 Generation Stats")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Prompt tokens", prompt_tokens)
    g2.metric("Completion tokens", completion_tokens)
    g3.metric("Total tokens", total_tokens)
    g4.metric("Latency", f"{generation_latency_ms:.0f} ms")
