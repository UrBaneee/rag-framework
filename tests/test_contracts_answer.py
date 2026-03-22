"""Unit tests for citation and answer contracts: Span, Citation, Answer, AnswerTrace."""

import pytest

from rag.core.contracts.answer import Answer
from rag.core.contracts.citation import Citation, Span, SpanType
from rag.core.contracts.trace import AnswerTrace, PipelineStep


@pytest.mark.unit
class TestSpan:
    def test_span_defaults(self):
        span = Span(text="Cloud deployment requires orchestration.")
        assert span.span_type == SpanType.ANSWER
        assert span.start == 0
        assert span.end == 0

    def test_span_citation_marker_type(self):
        span = Span(text="[1]", span_type=SpanType.CITATION_MARKER, start=40, end=43)
        assert span.span_type == SpanType.CITATION_MARKER
        assert span.start == 40

    def test_span_abstain_type(self):
        span = Span(text="I cannot answer based on the provided context.", span_type=SpanType.ABSTAIN)
        assert span.span_type == SpanType.ABSTAIN

    def test_span_type_enum_values(self):
        assert SpanType.ANSWER == "answer"
        assert SpanType.CITATION_MARKER == "citation_marker"
        assert SpanType.ABSTAIN == "abstain"
        assert SpanType.PREAMBLE == "preamble"


@pytest.mark.unit
class TestCitation:
    def test_citation_pdf_case(self):
        citation = Citation(
            ref_number=1,
            chunk_id="chk-001",
            doc_id="doc-001",
            source_label="architecture.pdf — page 12",
            page=12,
            display_text="All services are deployed on AWS/Azure.",
        )
        assert citation.ref_number == 1
        assert citation.page == 12
        assert "page 12" in citation.source_label

    def test_citation_chunk_only_case(self):
        citation = Citation(
            ref_number=2,
            chunk_id="chk-042",
            doc_id="doc-007",
            source_label="readme.md",
        )
        assert citation.page is None
        assert citation.display_text == ""

    def test_citation_ref_number_must_be_positive(self):
        with pytest.raises(Exception):
            Citation(ref_number=0, chunk_id="c", doc_id="d", source_label="x")


@pytest.mark.unit
class TestAnswer:
    def test_answer_with_citations(self):
        citation = Citation(
            ref_number=1,
            chunk_id="chk-001",
            doc_id="doc-001",
            source_label="architecture.pdf — page 12",
            page=12,
        )
        span_text = Span(text="Cloud deployment requires orchestration.", span_type=SpanType.ANSWER)
        span_ref = Span(text="[1]", span_type=SpanType.CITATION_MARKER)
        answer = Answer(
            text="Cloud deployment requires orchestration.[1]",
            citations=[citation],
            spans=[span_text, span_ref],
            query="How is deployment handled?",
        )
        assert len(answer.citations) == 1
        assert answer.citations[0].ref_number == 1
        assert len(answer.spans) == 2
        assert answer.abstained is False

    def test_answer_abstained(self):
        span = Span(
            text="I cannot answer based on the provided context.",
            span_type=SpanType.ABSTAIN,
        )
        answer = Answer(
            text="I cannot answer based on the provided context.",
            citations=[],
            spans=[span],
            abstained=True,
            query="What is the revenue forecast?",
        )
        assert answer.abstained is True
        assert answer.citations == []

    def test_answer_minimal(self):
        answer = Answer(text="The system uses FAISS for vector storage.")
        assert answer.text == "The system uses FAISS for vector storage."
        assert answer.citations == []
        assert answer.spans == []


@pytest.mark.unit
class TestAnswerTrace:
    def test_answer_trace_token_fields(self):
        trace = AnswerTrace(
            query="What is the deployment model?",
            prompt_tokens=512,
            completion_tokens=128,
            total_tokens=640,
            total_latency_ms=1250.5,
            model="gpt-4o-mini",
        )
        assert trace.prompt_tokens == 512
        assert trace.completion_tokens == 128
        assert trace.total_tokens == 640
        assert trace.total_latency_ms == 1250.5
        assert trace.model == "gpt-4o-mini"

    def test_answer_trace_with_steps(self):
        steps = [
            PipelineStep(step_name="bm25_retrieval", output_summary="20 candidates", latency_ms=12.3),
            PipelineStep(step_name="vector_retrieval", output_summary="20 candidates", latency_ms=45.1),
            PipelineStep(step_name="rrf_fusion", output_summary="40 fused", latency_ms=1.2),
            PipelineStep(step_name="reranker", output_summary="8 reranked", latency_ms=220.0),
        ]
        trace = AnswerTrace(
            query="test",
            steps=steps,
            candidates_before_rerank=40,
            candidates_after_rerank=8,
            context_chunks_used=3,
            rerank_provider="voyage",
        )
        assert len(trace.steps) == 4
        assert trace.candidates_before_rerank == 40
        assert trace.rerank_provider == "voyage"

    def test_answer_trace_defaults(self):
        trace = AnswerTrace(query="hello")
        assert trace.prompt_tokens == 0
        assert trace.total_latency_ms == 0.0
        assert trace.steps == []
        assert trace.run_id is None
