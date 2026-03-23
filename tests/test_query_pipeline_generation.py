"""Tests for QueryPipeline generation stage — Task 7.6."""

from unittest.mock import MagicMock, patch

import pytest

from rag.core.contracts.answer import Answer
from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.contracts.citation import Citation
from rag.core.contracts.trace import AnswerTrace
from rag.core.interfaces.llm_client import LLMResponse
from rag.infra.generation.answer_composer_basic import BasicAnswerComposer
from rag.pipelines.query_pipeline import QueryPipeline, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(chunk_id: str, text: str = "chunk text") -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id="doc.pdf",
        display_text=text,
        stable_text=text,
        bm25_score=1.0,
        rrf_score=0.5,
        final_score=0.5,
        retrieval_source=RetrievalSource.BM25,
        metadata={"source_label": "doc.pdf — page 1"},
    )


def _make_keyword_index(candidates: list[Candidate]) -> MagicMock:
    idx = MagicMock()
    idx.search.return_value = candidates
    return idx


def _make_trace_store() -> MagicMock:
    store = MagicMock()
    store.save_run.return_value = "run-test-001"
    return store


def _make_llm(answer_text: str = "RAG combines retrieval and generation.[1]") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(
        text=answer_text,
        model="gpt-4o-mini",
        prompt_tokens=40,
        completion_tokens=15,
        total_tokens=55,
        latency_ms=100.0,
    )
    return llm


def _make_pipeline(candidates, answer_text=None, with_composer=True):
    kw_idx = _make_keyword_index(candidates)
    trace_store = _make_trace_store()
    composer = None
    if with_composer:
        llm = _make_llm(answer_text or "Answer text.[1]")
        composer = BasicAnswerComposer(llm_client=llm, top_k=3)
    pipeline = QueryPipeline(
        keyword_index=kw_idx,
        trace_store=trace_store,
        answer_composer=composer,
    )
    return pipeline, trace_store


# ---------------------------------------------------------------------------
# Without composer — backwards compatibility
# ---------------------------------------------------------------------------


def test_query_without_composer_returns_result():
    cands = [_make_candidate("c0"), _make_candidate("c1")]
    pipeline, _ = _make_pipeline(cands, with_composer=False)
    result = pipeline.query("What is RAG?")
    assert isinstance(result, QueryResult)
    assert result.answer is None
    assert result.answer_trace is None


# ---------------------------------------------------------------------------
# With composer — Answer populated
# ---------------------------------------------------------------------------


def test_query_with_composer_returns_answer():
    cands = [_make_candidate("c0"), _make_candidate("c1")]
    pipeline, _ = _make_pipeline(cands)
    result = pipeline.query("What is RAG?")
    assert isinstance(result.answer, Answer)


def test_query_answer_text_populated():
    cands = [_make_candidate("c0")]
    pipeline, _ = _make_pipeline(cands, answer_text="RAG is retrieval augmented generation.[1]")
    result = pipeline.query("What is RAG?")
    assert result.answer.text == "RAG is retrieval augmented generation.[1]"


def test_query_answer_query_matches():
    cands = [_make_candidate("c0")]
    pipeline, _ = _make_pipeline(cands)
    result = pipeline.query("What is RAG?")
    assert result.answer.query == "What is RAG?"


# ---------------------------------------------------------------------------
# With composer — AnswerTrace populated
# ---------------------------------------------------------------------------


def test_query_answer_trace_populated():
    cands = [_make_candidate("c0")]
    pipeline, _ = _make_pipeline(cands)
    result = pipeline.query("What is RAG?")
    assert isinstance(result.answer_trace, AnswerTrace)


def test_query_answer_trace_token_counts():
    cands = [_make_candidate("c0")]
    pipeline, _ = _make_pipeline(cands)
    result = pipeline.query("What is RAG?")
    assert result.answer_trace.total_tokens == 55
    assert result.answer_trace.prompt_tokens == 40
    assert result.answer_trace.completion_tokens == 15


def test_query_answer_trace_model():
    cands = [_make_candidate("c0")]
    pipeline, _ = _make_pipeline(cands)
    result = pipeline.query("What is RAG?")
    assert result.answer_trace.model == "gpt-4o-mini"


def test_query_answer_trace_run_id():
    cands = [_make_candidate("c0")]
    pipeline, _ = _make_pipeline(cands)
    result = pipeline.query("What is RAG?")
    assert result.answer_trace.run_id is not None


# ---------------------------------------------------------------------------
# Generation stats stored in TraceStore
# ---------------------------------------------------------------------------


def test_query_generation_trace_stored():
    cands = [_make_candidate("c0")]
    pipeline, trace_store = _make_pipeline(cands)
    pipeline.query("What is RAG?")
    # Collect all save_run call types
    call_run_types = [
        call.kwargs.get("run_type") or call.args[0] if call.args else call.kwargs.get("run_type")
        for call in trace_store.save_run.call_args_list
    ]
    # Also check via keyword
    run_types = []
    for call in trace_store.save_run.call_args_list:
        kwargs = call.kwargs if call.kwargs else {}
        args = call.args if call.args else ()
        rt = kwargs.get("run_type") or (args[0] if args else None)
        run_types.append(rt)
    assert "query_generation" in run_types


def test_query_generation_trace_has_token_metadata():
    cands = [_make_candidate("c0")]
    pipeline, trace_store = _make_pipeline(cands)
    pipeline.query("What is RAG?")
    # Find the query_generation save_run call
    gen_call = None
    for call in trace_store.save_run.call_args_list:
        kwargs = call.kwargs if call.kwargs else {}
        args = call.args if call.args else ()
        rt = kwargs.get("run_type") or (args[0] if args else None)
        if rt == "query_generation":
            gen_call = call
            break
    assert gen_call is not None
    meta = gen_call.kwargs.get("metadata", {})
    assert "total_tokens" in meta
    assert meta["total_tokens"] == 55


# ---------------------------------------------------------------------------
# Citations from composer override pipeline citations
# ---------------------------------------------------------------------------


def test_query_citations_from_composer():
    cands = [_make_candidate("c0"), _make_candidate("c1")]
    pipeline, _ = _make_pipeline(cands, answer_text="Answer text.[1]")
    result = pipeline.query("What is RAG?")
    # Citations should reflect what the composer returned (only [1] referenced)
    assert len(result.citations) >= 1
    assert result.citations[0].ref_number == 1
