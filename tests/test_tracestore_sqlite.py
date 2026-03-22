"""Integration tests for SQLiteTraceStore write/read methods."""

import pytest

from rag.core.contracts.trace import AnswerTrace, PipelineStep
from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore


@pytest.fixture
def store(tmp_path):
    return SQLiteTraceStore(tmp_path / "trace.sqlite")


@pytest.fixture
def sample_trace():
    return AnswerTrace(
        query="What is the deployment model?",
        prompt_tokens=512,
        completion_tokens=128,
        total_tokens=640,
        total_latency_ms=1250.0,
        model="gpt-4o-mini",
        rerank_provider="voyage",
        candidates_before_rerank=40,
        candidates_after_rerank=8,
        context_chunks_used=3,
        steps=[
            PipelineStep(step_name="bm25_retrieval", output_summary="20 candidates", latency_ms=12.0),
            PipelineStep(step_name="vector_retrieval", output_summary="20 candidates", latency_ms=45.0),
            PipelineStep(step_name="rrf_fusion", output_summary="40 fused", latency_ms=1.0),
        ],
    )


@pytest.mark.integration
class TestRunCRUD:
    def test_save_run_returns_run_id(self, store):
        run_id = store.save_run("query", {"query": "hello"})
        assert isinstance(run_id, str)
        assert len(run_id) > 0

    def test_save_multiple_runs(self, store):
        id1 = store.save_run("query", {})
        id2 = store.save_run("ingest", {})
        assert id1 != id2

    def test_list_runs_all(self, store):
        store.save_run("query", {"q": "a"})
        store.save_run("ingest", {"path": "/tmp/f"})
        runs = store.list_runs()
        assert len(runs) == 2

    def test_list_runs_filtered_by_type(self, store):
        store.save_run("query", {})
        store.save_run("query", {})
        store.save_run("ingest", {})
        query_runs = store.list_runs(run_type="query")
        assert len(query_runs) == 2
        assert all(r["run_type"] == "query" for r in query_runs)

    def test_list_runs_limit(self, store):
        for i in range(5):
            store.save_run("query", {"i": i})
        runs = store.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_runs_metadata_preserved(self, store):
        store.save_run("ingest", {"source": "/docs/report.pdf", "pages": 42})
        runs = store.list_runs(run_type="ingest")
        assert runs[0]["metadata"]["source"] == "/docs/report.pdf"
        assert runs[0]["metadata"]["pages"] == 42


@pytest.mark.integration
class TestAnswerTraceCRUD:
    def test_save_and_get_answer_trace(self, store, sample_trace):
        run_id = store.save_run("query", {"query": sample_trace.query})
        store.save_answer_trace(run_id, sample_trace)

        retrieved = store.get_answer_trace(run_id)
        assert retrieved is not None
        assert retrieved.query == "What is the deployment model?"
        assert retrieved.prompt_tokens == 512
        assert retrieved.completion_tokens == 128
        assert retrieved.total_tokens == 640
        assert retrieved.total_latency_ms == 1250.0
        assert retrieved.model == "gpt-4o-mini"
        assert retrieved.rerank_provider == "voyage"
        assert retrieved.candidates_before_rerank == 40
        assert retrieved.context_chunks_used == 3
        assert retrieved.run_id == run_id

    def test_answer_trace_steps_preserved(self, store, sample_trace):
        run_id = store.save_run("query", {})
        store.save_answer_trace(run_id, sample_trace)
        retrieved = store.get_answer_trace(run_id)
        assert len(retrieved.steps) == 3
        assert retrieved.steps[0].step_name == "bm25_retrieval"
        assert retrieved.steps[2].step_name == "rrf_fusion"

    def test_get_answer_trace_missing_returns_none(self, store):
        assert store.get_answer_trace("nonexistent-run") is None

    def test_two_runs_independent_traces(self, store, sample_trace):
        run_id_1 = store.save_run("query", {})
        run_id_2 = store.save_run("query", {})
        store.save_answer_trace(run_id_1, sample_trace)

        assert store.get_answer_trace(run_id_1) is not None
        assert store.get_answer_trace(run_id_2) is None
