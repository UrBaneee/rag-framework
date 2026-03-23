"""Tests for MCP tool schemas — Task 9.1."""

import pytest
from pydantic import ValidationError

from rag.app.mcp_server.schemas import (
    CitationOutput,
    EvalRunToolInput,
    EvalRunToolOutput,
    IngestToolInput,
    IngestToolOutput,
    MetricResult,
    QueryToolInput,
    QueryToolOutput,
)


# ---------------------------------------------------------------------------
# rag.ingest — IngestToolInput
# ---------------------------------------------------------------------------


def test_ingest_input_valid_minimal():
    inp = IngestToolInput(source_path="/tmp/doc.pdf")
    assert inp.source_path == "/tmp/doc.pdf"
    assert inp.collection == "default"
    assert inp.token_budget == 512


def test_ingest_input_valid_full():
    inp = IngestToolInput(
        source_path="/data/report.pdf",
        collection="research",
        token_budget=256,
        embedding_provider="openai",
        embedding_model="text-embedding-3-small",
        vector_dimension=1536,
        db_path="data/rag.db",
        index_dir="data/indexes",
    )
    assert inp.embedding_provider == "openai"
    assert inp.vector_dimension == 1536


def test_ingest_input_blank_source_path_rejected():
    with pytest.raises(ValidationError):
        IngestToolInput(source_path="   ")


def test_ingest_input_empty_source_path_rejected():
    with pytest.raises(ValidationError):
        IngestToolInput(source_path="")


def test_ingest_input_token_budget_too_small():
    with pytest.raises(ValidationError):
        IngestToolInput(source_path="/tmp/f.txt", token_budget=32)


def test_ingest_input_token_budget_too_large():
    with pytest.raises(ValidationError):
        IngestToolInput(source_path="/tmp/f.txt", token_budget=9999)


# ---------------------------------------------------------------------------
# rag.ingest — IngestToolOutput
# ---------------------------------------------------------------------------


def test_ingest_output_valid():
    out = IngestToolOutput(
        doc_id="doc-abc",
        source_path="/tmp/doc.pdf",
        block_count=42,
        chunk_count=10,
        embed_tokens=1200,
        elapsed_ms=350.0,
    )
    assert out.error is None
    assert out.skipped is False


def test_ingest_output_skipped():
    out = IngestToolOutput(
        doc_id="doc-abc",
        source_path="/tmp/doc.pdf",
        block_count=0,
        chunk_count=0,
        elapsed_ms=5.0,
        skipped=True,
    )
    assert out.skipped is True


def test_ingest_output_with_error():
    out = IngestToolOutput(
        doc_id="",
        source_path="/tmp/bad.bin",
        block_count=0,
        chunk_count=0,
        elapsed_ms=2.0,
        error="Unsupported file type",
    )
    assert out.error == "Unsupported file type"


# ---------------------------------------------------------------------------
# rag.query — QueryToolInput
# ---------------------------------------------------------------------------


def test_query_input_valid_minimal():
    inp = QueryToolInput(query="What is RAG?")
    assert inp.query == "What is RAG?"
    assert inp.top_k == 10
    assert inp.enable_generation is True


def test_query_input_valid_full():
    inp = QueryToolInput(
        query="Explain retrieval augmented generation.",
        top_k=5,
        context_top_k=2,
        token_budget=1024,
        collection="research",
        embedding_provider="openai",
        llm_model="gpt-4o",
        enable_generation=True,
    )
    assert inp.llm_model == "gpt-4o"
    assert inp.context_top_k == 2


def test_query_input_blank_query_rejected():
    with pytest.raises(ValidationError):
        QueryToolInput(query="   ")


def test_query_input_empty_query_rejected():
    with pytest.raises(ValidationError):
        QueryToolInput(query="")


def test_query_input_top_k_out_of_range():
    with pytest.raises(ValidationError):
        QueryToolInput(query="Q", top_k=0)
    with pytest.raises(ValidationError):
        QueryToolInput(query="Q", top_k=101)


def test_query_input_no_generation():
    inp = QueryToolInput(query="What is RAG?", enable_generation=False)
    assert inp.enable_generation is False


# ---------------------------------------------------------------------------
# rag.query — QueryToolOutput
# ---------------------------------------------------------------------------


def test_query_output_valid_with_answer():
    out = QueryToolOutput(
        query="What is RAG?",
        answer="RAG is retrieval-augmented generation.[1]",
        citations=[
            CitationOutput(
                ref_number=1,
                chunk_id="c001",
                doc_id="doc.pdf",
                source_label="doc.pdf — page 1",
                display_text="RAG combines retrieval with generation.",
            )
        ],
        abstained=False,
        candidate_count=5,
        prompt_tokens=40,
        completion_tokens=15,
        total_tokens=55,
        generation_latency_ms=120.0,
        elapsed_ms=250.0,
    )
    assert len(out.citations) == 1
    assert out.citations[0].ref_number == 1


def test_query_output_abstained():
    out = QueryToolOutput(
        query="What is the meaning of life?",
        answer="I don't have enough information in the provided context.",
        abstained=True,
        elapsed_ms=100.0,
    )
    assert out.abstained is True


def test_query_output_with_error():
    out = QueryToolOutput(
        query="Q?",
        error="Index not found",
        elapsed_ms=2.0,
    )
    assert out.error == "Index not found"


def test_citation_output_valid():
    cit = CitationOutput(
        ref_number=2,
        chunk_id="chunk-xyz",
        doc_id="guide.pdf",
        source_label="guide.pdf — page 7",
        page=7,
        display_text="Key excerpt here.",
    )
    assert cit.ref_number == 2
    assert cit.page == 7


def test_citation_ref_number_must_be_positive():
    with pytest.raises(ValidationError):
        CitationOutput(ref_number=0, chunk_id="c", doc_id="d", source_label="s")


# ---------------------------------------------------------------------------
# rag.eval.run — EvalRunToolInput
# ---------------------------------------------------------------------------


def test_eval_input_valid_minimal():
    inp = EvalRunToolInput(dataset_path="/data/eval.jsonl")
    assert inp.dataset_path == "/data/eval.jsonl"
    assert "recall_at_k" in inp.metrics
    assert inp.top_k == 10


def test_eval_input_valid_full():
    inp = EvalRunToolInput(
        dataset_path="/data/eval.json",
        metrics=["recall_at_k", "mrr", "ndcg_at_k", "faithfulness", "answer_relevance"],
        top_k=5,
        collection="test",
        output_path="/tmp/report.json",
    )
    assert len(inp.metrics) == 5
    assert inp.output_path == "/tmp/report.json"


def test_eval_input_invalid_metric_rejected():
    with pytest.raises(ValidationError):
        EvalRunToolInput(dataset_path="/d.jsonl", metrics=["nonexistent_metric"])


def test_eval_input_empty_metrics_rejected():
    with pytest.raises(ValidationError):
        EvalRunToolInput(dataset_path="/d.jsonl", metrics=[])


# ---------------------------------------------------------------------------
# rag.eval.run — EvalRunToolOutput
# ---------------------------------------------------------------------------


def test_eval_output_valid():
    out = EvalRunToolOutput(
        dataset_path="/data/eval.jsonl",
        metrics=[
            MetricResult(metric="recall_at_k", value=0.82, num_samples=50),
            MetricResult(metric="mrr", value=0.75, num_samples=50),
        ],
        num_queries=50,
        elapsed_ms=3500.0,
    )
    assert len(out.metrics) == 2
    assert out.metrics[0].value == 0.82


def test_eval_output_with_error():
    out = EvalRunToolOutput(
        dataset_path="/data/eval.jsonl",
        error="Dataset file not found",
        elapsed_ms=1.0,
    )
    assert out.error == "Dataset file not found"


def test_metric_result_valid():
    m = MetricResult(metric="ndcg_at_k", value=0.91, num_samples=100)
    assert m.metric == "ndcg_at_k"
    assert m.value == 0.91
