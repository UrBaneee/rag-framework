"""Tests for BasicAnswerComposer — Task 7.5."""

from unittest.mock import MagicMock

import pytest

from rag.core.contracts.answer import Answer
from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.contracts.citation import SpanType
from rag.core.contracts.trace import AnswerTrace
from rag.core.interfaces.llm_client import LLMResponse
from rag.infra.generation.answer_composer_basic import BasicAnswerComposer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm(text: str = "RAG is retrieval-augmented generation.[1]",
              model: str = "gpt-4o-mini",
              prompt_tokens: int = 50,
              completion_tokens: int = 20) -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = LLMResponse(
        text=text,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=120.0,
    )
    return llm


def _make_candidate(chunk_id: str, text: str, doc_id: str = "doc.pdf") -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id=doc_id,
        display_text=text,
        stable_text=text,
        rrf_score=1.0,
        final_score=1.0,
        retrieval_source=RetrievalSource.BM25,
        metadata={"source_label": f"{doc_id} — page 1"},
    )


def _compose(answer_text: str = "RAG is retrieval-augmented generation.[1]",
             num_candidates: int = 2):
    llm = _mock_llm(text=answer_text)
    composer = BasicAnswerComposer(llm_client=llm, top_k=3)
    cands = [_make_candidate(f"c{i}", f"Context chunk {i}.") for i in range(num_candidates)]
    return composer.compose("What is RAG?", cands)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


def test_compose_returns_answer_and_trace():
    answer, trace = _compose()
    assert isinstance(answer, Answer)
    assert isinstance(trace, AnswerTrace)


# ---------------------------------------------------------------------------
# Answer fields
# ---------------------------------------------------------------------------


def test_answer_text_populated():
    answer, _ = _compose("RAG combines retrieval with generation.[1]")
    assert answer.text == "RAG combines retrieval with generation.[1]"


def test_answer_query_set():
    llm = _mock_llm()
    composer = BasicAnswerComposer(llm_client=llm)
    cands = [_make_candidate("c0", "Some context.")]
    answer, _ = composer.compose("What is RAG?", cands)
    assert answer.query == "What is RAG?"


def test_answer_not_abstained_for_normal_text():
    answer, _ = _compose("RAG stands for retrieval-augmented generation.[1]")
    assert answer.abstained is False


def test_answer_abstained_for_abstain_phrase():
    answer, _ = _compose(
        "I don't have enough information in the provided context to answer this question."
    )
    assert answer.abstained is True


def test_answer_abstained_case_insensitive():
    answer, _ = _compose(
        "I DO NOT HAVE ENOUGH INFORMATION IN THE PROVIDED CONTEXT to answer."
    )
    assert answer.abstained is True


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_answer_contains_citations():
    answer, _ = _compose("RAG uses retrieval.[1]")
    assert len(answer.citations) >= 1


def test_citations_are_one_based():
    answer, _ = _compose("See context.[1][2]", num_candidates=3)
    ref_numbers = [c.ref_number for c in answer.citations]
    assert 1 in ref_numbers
    assert 2 in ref_numbers


def test_unused_citation_filtered_out():
    # Only [1] referenced — [2] should be filtered
    llm = _mock_llm(text="Only first source.[1]")
    composer = BasicAnswerComposer(llm_client=llm, top_k=3)
    cands = [
        _make_candidate("c0", "First chunk."),
        _make_candidate("c1", "Second chunk."),
    ]
    answer, _ = composer.compose("Question?", cands)
    ref_numbers = {c.ref_number for c in answer.citations}
    assert 1 in ref_numbers
    assert 2 not in ref_numbers


def test_all_citations_kept_when_no_inline_markers():
    # LLM returns answer without any [N] markers
    answer, _ = _compose("RAG is a useful technique.")
    # All packed citations should be preserved (no markers = keep all)
    assert len(answer.citations) >= 1


# ---------------------------------------------------------------------------
# Spans
# ---------------------------------------------------------------------------


def test_spans_contain_citation_markers():
    answer, _ = _compose("Some answer text.[1]")
    marker_spans = [s for s in answer.spans if s.span_type == SpanType.CITATION_MARKER]
    assert len(marker_spans) >= 1


def test_spans_contain_answer_text():
    answer, _ = _compose("Some answer text.[1]")
    answer_spans = [s for s in answer.spans if s.span_type == SpanType.ANSWER]
    assert len(answer_spans) >= 1


def test_spans_cover_full_text():
    answer, _ = _compose("First part.[1] Second part.[2]")
    reconstructed = "".join(s.text for s in answer.spans)
    assert reconstructed == answer.text


# ---------------------------------------------------------------------------
# AnswerTrace fields
# ---------------------------------------------------------------------------


def test_trace_query_set():
    _, trace = _compose()
    assert trace.query == "What is RAG?"


def test_trace_token_counts():
    _, trace = _compose()
    assert trace.prompt_tokens == 50
    assert trace.completion_tokens == 20
    assert trace.total_tokens == 70


def test_trace_model_set():
    _, trace = _compose()
    assert trace.model == "gpt-4o-mini"


def test_trace_context_chunks_used():
    _, trace = _compose(num_candidates=2)
    assert trace.context_chunks_used == 2


def test_trace_has_pipeline_steps():
    _, trace = _compose()
    step_names = [s.step_name for s in trace.steps]
    assert "context_pack" in step_names
    assert "prompt_build" in step_names
    assert "llm_generate" in step_names


def test_trace_run_id_forwarded():
    llm = _mock_llm()
    composer = BasicAnswerComposer(llm_client=llm)
    cands = [_make_candidate("c0", "Context.")]
    _, trace = composer.compose("Q?", cands, run_id="run-abc")
    assert trace.run_id == "run-abc"


def test_trace_total_latency_positive():
    _, trace = _compose()
    assert trace.total_latency_ms > 0
