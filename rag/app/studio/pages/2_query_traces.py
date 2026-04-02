"""Query Traces page — submit queries and inspect the full retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Query Traces · RAG Studio", layout="wide")

st.title("💬 Query Traces")
st.caption("Submit a query and inspect retrieval, reranking, context packing, and generation.")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "query_result" not in st.session_state:
    st.session_state.query_result = None
if "query_running" not in st.session_state:
    st.session_state.query_running = False
if "query_error" not in st.session_state:
    st.session_state.query_error = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---------------------------------------------------------------------------
# Sidebar — pipeline configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")
    db_path = st.text_input("SQLite DB path", value="data/rag.db")
    index_dir = st.text_input("Index directory", value="data/indexes")

    st.divider()
    st.subheader("Retrieval")
    top_k = st.number_input("top_k per index", min_value=1, max_value=50, value=10)

    st.subheader("Embedding")
    embedding_provider = st.selectbox(
        "Provider", ["openai", "multilingual", "none"], index=0,
        help=(
            "**openai** — OpenAI API (requires OPENAI_API_KEY).\n\n"
            "**multilingual** — Local sentence-transformers, no API key, "
            "cross-lingual retrieval.\n\n"
            "**none** — BM25 only."
        ),
    )
    _ml_model = "paraphrase-multilingual-mpnet-base-v2"
    _ml_dim = 768
    embedding_model = st.text_input(
        "Model",
        value=_ml_model if embedding_provider == "multilingual" else "text-embedding-3-small",
        disabled=(embedding_provider == "none"),
    )
    vector_dimension = st.number_input(
        "Vector dimension", min_value=1, max_value=4096,
        value=_ml_dim if embedding_provider == "multilingual" else 1536,
        disabled=(embedding_provider == "none"),
    )

    st.subheader("Context Packing")
    context_top_k = st.number_input("context_top_k", min_value=1, max_value=20, value=6)
    token_budget = st.number_input("token_budget", min_value=64, max_value=4096, value=2048)

    st.subheader("Reranker")
    reranker_provider = st.selectbox(
        "Provider",
        ["none", "crossencoder", "voyage"],
        index=0,
        help=(
            "**none** — No reranking; RRF score is the final score.\n\n"
            "**crossencoder** — Local cross-encoder (sentence-transformers). "
            "No API key required. Downloads ~80 MB model on first use.\n\n"
            "**voyage** — Voyage AI reranker API (requires VOYAGE_API_KEY)."
        ),
    )
    _default_reranker_model = {
        "crossencoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "voyage": "rerank-2",
    }.get(reranker_provider, "")
    reranker_model = st.text_input(
        "Reranker model",
        value=_default_reranker_model,
        disabled=(reranker_provider == "none"),
        help="Model name for the selected reranker.",
    )

    st.subheader("Generation")
    llm_model = st.text_input("LLM model", value="gpt-4o-mini")
    enable_generation = st.checkbox("Enable generation", value=True)

# ---------------------------------------------------------------------------
# Query input + submit
# ---------------------------------------------------------------------------

st.markdown("### 🔎 Query")

# Form provides Enter-to-submit behaviour for free
with st.form(key="query_form", clear_on_submit=False):
    query_text = st.text_area(
        "Enter your question",
        placeholder="e.g. What is retrieval-augmented generation?",
        height=80,
        help="Press Ctrl+Enter or click Run Query to submit.",
    )
    submitted = st.form_submit_button(
        "▶ Run Query",
        type="primary",
        disabled=st.session_state.query_running,
    )

# ---------------------------------------------------------------------------
# Validate + execute on submit
# ---------------------------------------------------------------------------

if submitted:
    if not query_text or not query_text.strip():
        st.error("⚠️ Please enter a non-empty query before submitting.")
    else:
        st.session_state.query_running = True
        st.session_state.query_result = None
        st.session_state.query_error = None
        st.session_state.last_query = query_text.strip()

        with st.spinner("Running query pipeline…"):
            try:
                from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
                from rag.infra.stores.docstore_sqlite import SQLiteDocStore
                from rag.infra.indexes.bm25_local import BM25LocalIndex
                from rag.infra.indexes.faiss_local import FaissLocalIndex
                from rag.infra.indexes.index_manager import IndexManager
                from rag.pipelines.query_pipeline import QueryPipeline

                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
                Path(index_dir).mkdir(parents=True, exist_ok=True)

                trace_store = SQLiteTraceStore(db_path)
                mgr = IndexManager(index_dir=index_dir)

                embed_provider = None
                vec_index = None
                if embedding_provider == "openai":
                    from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
                    embed_provider = OpenAIEmbeddingProvider(
                        model=embedding_model,
                        dimensions=int(vector_dimension),
                    )
                    vec_index = mgr.faiss
                elif embedding_provider == "multilingual":
                    from rag.infra.embedding.multilingual_embedding import MultilingualEmbeddingProvider
                    embed_provider = MultilingualEmbeddingProvider(
                        model=embedding_model,
                        dim=int(vector_dimension),
                    )
                    vec_index = mgr.faiss

                composer = None
                if enable_generation:
                    from rag.infra.llm.openai_llm_client import OpenAILLMClient
                    from rag.infra.generation.answer_composer_basic import BasicAnswerComposer
                    llm = OpenAILLMClient(model=llm_model)
                    composer = BasicAnswerComposer(
                        llm_client=llm,
                        top_k=int(context_top_k),
                        token_budget=int(token_budget),
                    )

                reranker = None
                if reranker_provider == "crossencoder":
                    from rag.infra.rerank.crossencoder_reranker import CrossEncoderReranker
                    reranker = CrossEncoderReranker(model=reranker_model)
                elif reranker_provider == "voyage":
                    from rag.infra.rerank.voyage_rerank import VoyageReranker
                    reranker = VoyageReranker(model=reranker_model)

                pipeline = QueryPipeline(
                    keyword_index=mgr.bm25,
                    trace_store=trace_store,
                    vector_index=vec_index,
                    embedding_provider=embed_provider,
                    answer_composer=composer,
                    reranker=reranker,
                    top_k=int(top_k),
                )

                result = pipeline.query(st.session_state.last_query)
                st.session_state.query_result = result

            except Exception as exc:
                st.session_state.query_error = str(exc)

        st.session_state.query_running = False
        st.rerun()

# ---------------------------------------------------------------------------
# Display error
# ---------------------------------------------------------------------------

if st.session_state.query_error:
    st.error(f"❌ Query failed: {st.session_state.query_error}")

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

result = st.session_state.query_result
if result is not None:
    st.divider()
    st.markdown(f"**Query:** {st.session_state.last_query}")
    st.caption(f"Run ID: `{result.run_id}` | Elapsed: {result.elapsed_ms:.0f} ms")

    from rag.app.studio.components.candidate_table import (
        render_candidate_table,
        render_context_packing_details,
        render_answer_section,
    )

    # Convert Pydantic models → plain dicts for the component
    def _cand_to_dict(c) -> dict:
        d = c.model_dump() if hasattr(c, "model_dump") else dict(c)
        if hasattr(c, "retrieval_source"):
            d["retrieval_source"] = c.retrieval_source.value if hasattr(c.retrieval_source, "value") else str(c.retrieval_source)
        return d

    cand_dicts = [_cand_to_dict(c) for c in result.candidates]

    # Retrieved chunks
    render_candidate_table(cand_dicts, title="📋 Retrieved & Ranked Chunks")

    st.divider()

    # Context packing details (derived from answer_trace if available)
    if result.answer_trace:
        at = result.answer_trace
        packed_count = at.context_chunks_used
        packed_cands = cand_dicts[:packed_count]
        # Estimate packed tokens and truncated from AnswerTrace steps
        pack_step = next((s for s in at.steps if s.step_name == "context_pack"), None)
        packed_tokens = 0
        truncated = False
        if pack_step:
            summary = pack_step.output_summary
            # Parse "N chunks packed, M tokens (truncated)"
            import re
            m = re.search(r"(\d+) tokens", summary)
            if m:
                packed_tokens = int(m.group(1))
            truncated = "(truncated)" in summary

        render_context_packing_details(
            packed_candidates=packed_cands,
            all_candidates=cand_dicts,
            context_top_k=int(context_top_k),
            token_budget=int(token_budget),
            packed_tokens=packed_tokens,
            truncated=truncated,
        )
        st.divider()

        # Answer + citations + generation stats
        cit_dicts = []
        if result.answer and result.answer.citations:
            cit_dicts = [c.model_dump() for c in result.answer.citations]
        elif result.citations:
            cit_dicts = [c.model_dump() for c in result.citations]

        render_answer_section(
            answer_text=result.answer.text if result.answer else "",
            citations=cit_dicts,
            abstained=result.answer.abstained if result.answer else False,
            prompt_tokens=at.prompt_tokens,
            completion_tokens=at.completion_tokens,
            total_tokens=at.total_tokens,
            generation_latency_ms=at.total_latency_ms,
        )
    else:
        # No generation — show citations from retrieval
        if result.citations:
            st.subheader("📎 Citations")
            for cit in result.citations:
                st.markdown(f"[{cit.ref_number}] {cit.source_label}")
