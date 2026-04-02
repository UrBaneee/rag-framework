"""Evaluation Panel — run and review RAG evaluation suites.

Supports:
- example_queries suite (representative grounded questions)
- failure_cases suite (ambiguous / out-of-domain / noisy queries)
- resume_qrels suite (non-circular human-labeled gold eval on resume corpus)

Metrics displayed: Recall@K, MRR, nDCG@K, source attribution,
efficiency (latency, token savings).  Tooltips come from
``metric_glossary.py``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st

# Ensure .env is loaded — RAGAS creates its own OpenAI client and needs the key
load_dotenv(override=False)

st.set_page_config(page_title="Evaluation Panel · RAG Studio", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent.parent.parent.parent.parent / "tests" / "fixtures"

_SUITES = {
    "example_queries": _FIXTURES_DIR / "example_queries.json",
    "failure_cases":   _FIXTURES_DIR / "failure_cases.json",
    "resume_qrels":    _FIXTURES_DIR / "resume_qrels.json",
}

# Metadata shown in the UI for each suite
_SUITE_META = {
    "example_queries": {
        "label":       "📋 Example Queries (representative)",
        "type":        "Regression suite — RAG documentation corpus",
        "status":      "⚠️ Circular (ground truth auto-labeled from BM25 top-3)",
        "limitation":  None,
    },
    "failure_cases": {
        "label":       "⚠️ Failure Cases (challenging)",
        "type":        "Adversarial suite — ambiguous / out-of-domain / noisy queries",
        "status":      "ℹ️ Requires manual expected_sources to be meaningful",
        "limitation":  None,
    },
    "resume_qrels": {
        "label":       "🏅 Resume Gold Eval (human-labeled)",
        "type":        "Gold eval on current ingested resume corpus",
        "status":      "✅ Non-circular — ground truth written from document content",
        "limitation":  (
            "⚠️ **Bound to current chunk IDs.** "
            "Scores will drop to 0 after re-ingestion or chunking/parser changes. "
            "Re-run the index rebuild script and update `resume_qrels.json` "
            "chunk IDs whenever the corpus changes."
        ),
    },
}

_DEFAULT_DB = "data/rag.db"
_DEFAULT_INDEX = "data/indexes"

_EMBEDDING_PROVIDERS = ["openai", "multilingual", "none"]
_ML_MODEL = "paraphrase-multilingual-mpnet-base-v2"
_ML_DIM = 768


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "eval_report": None,
        "eval_suite": None,
        "eval_running": False,
        "eval_error": None,
        "ragas_result": None,
        "ragas_error": None,
        "ragas_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_suite(path: Path) -> list[dict]:
    """Load a JSON or JSONL suite file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _run_evaluation(
    suite_name: str,
    db_path: str,
    index_dir: str,
    top_k: int,
    k: int,
    embedding_provider: str = "none",
    embedding_model: str = "",
    vector_dimension: int = 1536,
    collection: str | None = None,
) -> tuple:
    """Load suite, run queries, compute metrics.

    Returns:
        (EvalReport, list[dict] raw_results) or raises on error.
    """
    from rag.infra.indexes.index_manager import IndexManager
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.eval_pipeline import run_eval
    from rag.pipelines.query_pipeline import QueryPipeline

    suite_path = _SUITES[suite_name]
    entries = _load_suite(suite_path)

    trace_store = SQLiteTraceStore(db_path)
    manager = IndexManager(index_dir)

    # Configure embedding provider for vector retrieval
    embed_provider = None
    vec_index = None
    if embedding_provider == "openai":
        from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
        embed_provider = OpenAIEmbeddingProvider(
            model=embedding_model,
            dimensions=int(vector_dimension),
        )
        vec_index = manager.faiss
    elif embedding_provider == "multilingual":
        from rag.infra.embedding.multilingual_embedding import MultilingualEmbeddingProvider
        embed_provider = MultilingualEmbeddingProvider(
            model=embedding_model,
            dim=int(vector_dimension),
        )
        vec_index = manager.faiss

    pipeline = QueryPipeline(
        keyword_index=manager.bm25,
        vector_index=vec_index,
        embedding_provider=embed_provider,
        trace_store=trace_store,
        top_k=top_k,
        answer_composer=None,
    )

    raw_results = []
    for entry in entries:
        query = entry.get("query", "")
        expected_sources = entry.get("expected_sources", [])
        expected_behavior = entry.get("expected_behavior", "")
        query_id = entry.get("query_id", query[:40])

        # Skip unanswerable queries — they have no expected_sources and
        # would drag Recall to 0 unfairly.  Record them as skipped so
        # the per-case table can still display them.
        if expected_behavior == "abstain":
            raw_results.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "retrieved": [],
                    "relevant": [],
                    "candidates": [],
                    "expected_behavior": "abstain",
                    "query_latency_ms": 0.0,
                    "error": None,
                    "skipped": True,
                }
            )
            continue

        t0 = time.perf_counter()
        qr = pipeline.query(query, collection=collection)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        raw_results.append(
            {
                "query_id": query_id,
                "query": query,
                "retrieved": [c.chunk_id for c in qr.candidates],
                "relevant": expected_sources,
                "candidates": qr.candidates,
                "expected_behavior": expected_behavior,
                "query_latency_ms": elapsed_ms,
                "error": qr.error,
            }
        )

    # Only pass non-skipped entries to the metric computation
    eval_results = [r for r in raw_results if not r.get("skipped")]
    report = run_eval(eval_results, k=k)
    return report, raw_results


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.title("📊 Evaluation Panel")
st.caption("Run evaluation suites and review RAG quality metrics.")

tab_run, tab_results, tab_ragas, tab_glossary = st.tabs(
    ["▶ Run Evaluation", "📈 Results", "🧪 Answer Quality (RAGAS)", "📖 Metric Glossary"]
)


# ---------------------------------------------------------------------------
# Tab 1 — Run Evaluation
# ---------------------------------------------------------------------------

with tab_run:
    st.subheader("Configuration")

    with st.form("eval_config_form"):
        col_l, col_r = st.columns(2)

        with col_l:
            suite_choice = st.selectbox(
                "Evaluation suite",
                options=list(_SUITES.keys()),
                format_func=lambda x: _SUITE_META[x]["label"],
            )
            top_k = st.number_input(
                "Top-K candidates to retrieve", min_value=1, max_value=50, value=10
            )
            embedding_provider = st.selectbox(
                "Embedding provider",
                options=_EMBEDDING_PROVIDERS,
                index=0,
                help=(
                    "**openai** — uses OpenAI API (needs OPENAI_API_KEY).\n\n"
                    "**multilingual** — local sentence-transformers, no API key.\n\n"
                    "**none** — BM25 only (no vector retrieval)."
                ),
            )

        with col_r:
            db_path = st.text_input("Database path", value=_DEFAULT_DB)
            index_dir = st.text_input("Index directory", value=_DEFAULT_INDEX)
            k_cutoff = st.number_input(
                "Metric cut-off (K)", min_value=1, max_value=50, value=10
            )
            _default_emb_model = (
                _ML_MODEL if embedding_provider == "multilingual"
                else "text-embedding-3-small"
            )
            _default_emb_dim = (
                _ML_DIM if embedding_provider == "multilingual" else 1536
            )
            embedding_model = st.text_input(
                "Embedding model",
                value=_default_emb_model,
                disabled=(embedding_provider == "none"),
            )
            vector_dimension = st.number_input(
                "Vector dimension",
                min_value=1, max_value=4096,
                value=_default_emb_dim,
                disabled=(embedding_provider == "none"),
            )

        submitted = st.form_submit_button(
            "▶ Run Evaluation",
            disabled=st.session_state.eval_running,
            use_container_width=True,
        )

    # Suite info card — shown immediately below the form
    _meta = _SUITE_META.get(suite_choice, {})
    with st.container():
        info_col, _ = st.columns([3, 1])
        with info_col:
            st.markdown(
                f"**Type:** {_meta.get('type', '—')}  \n"
                f"**Status:** {_meta.get('status', '—')}"
            )
            if _meta.get("limitation"):
                st.warning(_meta["limitation"])

    if submitted:
        db = Path(db_path)
        if not db.exists():
            st.error(f"Database not found: `{db_path}`. Run ingestion first.")
        else:
            st.session_state.eval_running = True
            st.session_state.eval_error = None
            st.session_state.eval_report = None

            with st.spinner(f"Running `{suite_choice}` suite…"):
                try:
                    # Auto-scope resume eval to its own collection so the
                    # RAG textbook chunks don't dominate retrieval.
                    _collection = "resumes" if suite_choice == "resume_qrels" else None
                    report, _ = _run_evaluation(
                        suite_name=suite_choice,
                        db_path=db_path,
                        index_dir=index_dir,
                        top_k=int(top_k),
                        k=int(k_cutoff),
                        embedding_provider=embedding_provider,
                        embedding_model=embedding_model,
                        vector_dimension=int(vector_dimension),
                        collection=_collection,
                    )
                    st.session_state.eval_report = report
                    st.session_state.eval_suite = suite_choice
                    st.session_state.eval_running = False
                    st.success(
                        f"Evaluation complete — {report.num_queries} queries "
                        f"evaluated.  Switch to the **Results** tab to review."
                    )
                except Exception as exc:
                    st.session_state.eval_running = False
                    st.session_state.eval_error = str(exc)
                    st.error(f"Evaluation failed: {exc}")

    # Suite preview
    with st.expander("Preview suite entries", expanded=False):
        preview_suite = st.selectbox(
            "Suite to preview",
            options=list(_SUITES.keys()),
            key="preview_suite_select",
        )
        suite_file = _SUITES.get(str(preview_suite))
        if suite_file is not None and suite_file.exists():
            entries = _load_suite(suite_file)
            st.write(f"**{len(entries)} entries** in `{suite_file.name}`")
            for i, e in enumerate(entries[:5]):
                with st.expander(
                    f"[{e.get('query_id', i+1)}] {e.get('query', '')[:80]}",
                    expanded=False,
                ):
                    st.json(e)
            if len(entries) > 5:
                st.caption(f"… and {len(entries) - 5} more entries.")
        else:
            st.warning(f"Suite file not found: `{suite_file}`")


# ---------------------------------------------------------------------------
# Tab 2 — Results
# ---------------------------------------------------------------------------

with tab_results:
    report = st.session_state.eval_report

    if report is None:
        st.info("No evaluation results yet. Run an evaluation in the **▶ Run Evaluation** tab.")
    else:
        from rag.app.studio.components.metrics_table import (
            render_efficiency_metrics,
            render_metrics_summary,
            render_per_case_table,
            render_source_attribution,
        )

        _res_suite = st.session_state.eval_suite or ""
        _res_meta  = _SUITE_META.get(_res_suite, {})
        suite_label = _res_meta.get("label", _res_suite)

        st.markdown(
            f"**Suite:** {suite_label} &nbsp;·&nbsp; "
            f"**Queries:** {report.num_queries} &nbsp;·&nbsp; "
            f"**K:** {report.k}"
        )
        if _res_meta.get("limitation"):
            st.warning(_res_meta["limitation"])
        st.divider()

        render_metrics_summary(report)
        st.divider()

        render_source_attribution(report)
        st.divider()

        render_efficiency_metrics(report)
        st.divider()

        # Answer quality (RAGAS) — Task 14.3
        from rag.app.studio.components.metrics_table import render_answer_quality
        render_answer_quality(report.ragas_metrics)
        if report.ragas_metrics is not None:
            st.divider()

        render_per_case_table(report.per_query, k=report.k)


# ---------------------------------------------------------------------------
# Tab 3 — Answer Quality (RAGAS)
# ---------------------------------------------------------------------------

with tab_ragas:
    st.subheader("Answer Quality (RAGAS)")
    st.caption(
        "Measures **faithfulness**, **answer relevancy**, and **context precision** "
        "using the golden test set (`tests/fixtures/golden_answer_set.json`).  "
        "Requires LLM generation — each query makes one API call."
    )

    _golden_path = _FIXTURES_DIR / "golden_answer_set.json"
    if not _golden_path.exists():
        st.error(f"Golden test set not found: `{_golden_path}`")
    else:
        _golden_entries = _load_suite(_golden_path)
        st.info(
            f"**{len(_golden_entries)} queries** in golden test set.  "
            "Expected answers were written manually — results are non-circular."
        )

        with st.form("ragas_form"):
            rc_l, rc_r = st.columns(2)
            with rc_l:
                ragas_db     = st.text_input("Database path", value=_DEFAULT_DB, key="ragas_db")
                ragas_index  = st.text_input("Index directory", value=_DEFAULT_INDEX, key="ragas_idx")
                ragas_emb    = st.selectbox(
                    "Embedding provider", _EMBEDDING_PROVIDERS, index=0, key="ragas_emb"
                )
            with rc_r:
                ragas_llm    = st.text_input("LLM model", value="gpt-4o-mini", key="ragas_llm")
                ragas_top_k  = st.number_input(
                    "context_top_k", min_value=1, max_value=20, value=6, key="ragas_topk"
                )
                ragas_budget = st.number_input(
                    "token_budget", min_value=64, max_value=4096, value=2048, key="ragas_budget"
                )
            ragas_submitted = st.form_submit_button(
                "▶ Run RAGAS Evaluation",
                disabled=st.session_state.ragas_running,
                use_container_width=True,
            )

        if ragas_submitted:
            st.session_state.ragas_running = True
            st.session_state.ragas_result  = None
            st.session_state.ragas_error   = None

            with st.spinner(f"Running RAGAS on {len(_golden_entries)} queries…"):
                try:
                    from rag.infra.indexes.index_manager import IndexManager
                    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
                    from rag.infra.llm.openai_llm_client import OpenAILLMClient
                    from rag.infra.generation.answer_composer_basic import BasicAnswerComposer
                    from rag.infra.evaluation.ragas_evaluator import RagasEvaluator
                    from rag.pipelines.query_pipeline import QueryPipeline
                    from rag.pipelines.eval_pipeline import run_golden_eval

                    mgr = IndexManager(ragas_index)
                    trace_store = SQLiteTraceStore(ragas_db)

                    r_embed = None
                    r_vec   = None
                    if ragas_emb == "openai":
                        from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
                        r_embed = OpenAIEmbeddingProvider(
                            model="text-embedding-3-small", dimensions=1536
                        )
                        r_vec = mgr.faiss
                    elif ragas_emb == "multilingual":
                        from rag.infra.embedding.multilingual_embedding import MultilingualEmbeddingProvider
                        r_embed = MultilingualEmbeddingProvider(
                            model=_ML_MODEL, dim=_ML_DIM
                        )
                        r_vec = mgr.faiss

                    llm       = OpenAILLMClient(model=ragas_llm)
                    composer  = BasicAnswerComposer(
                        llm_client=llm,
                        top_k=int(ragas_top_k),
                        token_budget=int(ragas_budget),
                    )
                    pipeline  = QueryPipeline(
                        keyword_index=mgr.bm25,
                        vector_index=r_vec,
                        embedding_provider=r_embed,
                        trace_store=trace_store,
                        answer_composer=composer,
                        top_k=int(ragas_top_k),
                    )
                    evaluator = RagasEvaluator()
                    metrics   = run_golden_eval(
                        golden_entries=_golden_entries,
                        query_pipeline=pipeline,
                        evaluator=evaluator,
                        trace_store=trace_store,
                    )
                    st.session_state.ragas_result = metrics
                except Exception as exc:
                    st.session_state.ragas_error = str(exc)

            st.session_state.ragas_running = False
            st.rerun()

        if st.session_state.ragas_error:
            st.error(f"❌ RAGAS evaluation failed: {st.session_state.ragas_error}")

        if st.session_state.ragas_result is not None:
            from rag.app.studio.components.metrics_table import render_answer_quality
            render_answer_quality(st.session_state.ragas_result)

            r = st.session_state.ragas_result
            if r.ragas_available and r.num_evaluated > 0:
                st.divider()
                st.caption(
                    f"Evaluated {r.num_evaluated} / {len(_golden_entries)} queries.  "
                    f"Skipped: {len(_golden_entries) - r.num_evaluated}."
                )


# ---------------------------------------------------------------------------
# Tab 4 — Metric Glossary
# ---------------------------------------------------------------------------

with tab_glossary:
    from rag.app.studio.components.metric_glossary import GLOSSARY

    st.subheader("Metric Glossary")
    st.caption("Definitions, direction, and common pitfalls for every metric shown in this panel.")

    _SECTIONS = {
        "Retrieval Metrics": ["recall_at_k", "mrr", "ndcg_at_k"],
        "Source Attribution": ["bm25_only", "vector_only", "both"],
        "Efficiency Metrics": [
            "token_saved_est",
            "mean_query_latency_ms",
            "mean_ingest_latency_ms",
            "skipped_chunks",
            "changed_chunks",
        ],
        "Answer Quality (RAGAS)": [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
        ],
    }

    _DIRECTION_LABEL = {
        "higher_is_better": "↑ Higher is better",
        "lower_is_better": "↓ Lower is better",
        "neutral": "→ Context-dependent",
    }

    for section_title, keys in _SECTIONS.items():
        st.markdown(f"### {section_title}")
        for key in keys:
            entry = GLOSSARY.get(key)
            if not entry:
                continue
            with st.expander(f"**{entry['label']}** — `{key}`", expanded=False):
                st.markdown(entry["description"])
                dir_text = _DIRECTION_LABEL.get(entry["direction"], entry["direction"])
                st.markdown(f"**{dir_text}**")
                st.warning(f"⚠️ {entry['pitfalls']}")
        st.divider()
