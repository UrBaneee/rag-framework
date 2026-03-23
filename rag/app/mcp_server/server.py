"""MCP server — exposes rag.ingest, rag.query, and rag.eval.run tools."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

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
from rag.app.mcp_server.wiring import build_ingest_pipeline, build_query_pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="rag-framework",
    instructions=(
        "RAG Framework MCP server. Provides tools for document ingestion, "
        "grounded question answering, and retrieval evaluation."
    ),
)


# ---------------------------------------------------------------------------
# rag.ingest
# ---------------------------------------------------------------------------


@mcp.tool(
    name="rag.ingest",
    description=(
        "Ingest a document into the RAG corpus. "
        "Parses, cleans, chunks, and optionally embeds the file, "
        "then stores it in the DocStore and indexes."
    ),
)
def rag_ingest(input: IngestToolInput) -> IngestToolOutput:
    """Ingest a single document file.

    Args:
        input: Validated IngestToolInput payload.

    Returns:
        IngestToolOutput with doc_id, chunk_count, elapsed_ms, and error.
    """
    logger.info("rag.ingest called: source_path=%s", input.source_path)
    try:
        pipeline = build_ingest_pipeline(
            db_path=input.db_path,
            index_dir=input.index_dir,
            token_budget=input.token_budget,
            embedding_provider=input.embedding_provider,
            embedding_model=input.embedding_model,
            vector_dimension=input.vector_dimension,
        )
        result = pipeline.ingest(input.source_path)

        # Persist indexes if they were updated
        if input.embedding_provider == "openai":
            from rag.infra.indexes.index_manager import IndexManager
            mgr = IndexManager(index_dir=input.index_dir)
            mgr.save()

        return IngestToolOutput(
            doc_id=result.doc_id,
            source_path=result.source_path,
            block_count=result.block_count,
            chunk_count=result.chunk_count,
            embed_tokens=result.embed_tokens,
            elapsed_ms=result.elapsed_ms,
            skipped=result.skipped,
            error=result.error,
            run_id=result.run_id,
        )
    except Exception as exc:
        logger.exception("rag.ingest failed: %s", exc)
        return IngestToolOutput(
            doc_id="",
            source_path=input.source_path,
            block_count=0,
            chunk_count=0,
            elapsed_ms=0.0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# rag.query
# ---------------------------------------------------------------------------


@mcp.tool(
    name="rag.query",
    description=(
        "Query the RAG corpus with a natural-language question. "
        "Retrieves relevant chunks via BM25 and optional vector search, "
        "applies RRF fusion, and optionally generates a grounded answer."
    ),
)
def rag_query(input: QueryToolInput) -> QueryToolOutput:
    """Answer a question against the ingested corpus.

    Args:
        input: Validated QueryToolInput payload.

    Returns:
        QueryToolOutput with answer text, citations, and generation stats.
    """
    logger.info("rag.query called: query=%r", input.query)
    try:
        pipeline = build_query_pipeline(
            db_path=input.db_path,
            index_dir=input.index_dir,
            top_k=input.top_k,
            context_top_k=input.context_top_k,
            token_budget=input.token_budget,
            embedding_provider=input.embedding_provider,
            embedding_model=input.embedding_model,
            vector_dimension=input.vector_dimension,
            llm_model=input.llm_model,
            enable_generation=input.enable_generation,
        )
        result = pipeline.query(input.query)

        # Extract answer fields
        answer_text = ""
        abstained = False
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        generation_latency_ms = 0.0

        if result.answer:
            answer_text = result.answer.text
            abstained = result.answer.abstained

        if result.answer_trace:
            at = result.answer_trace
            prompt_tokens = at.prompt_tokens
            completion_tokens = at.completion_tokens
            total_tokens = at.total_tokens
            generation_latency_ms = at.total_latency_ms

        # Build citation outputs
        citations_out: list[CitationOutput] = []
        source_citations = result.answer.citations if result.answer else result.citations
        for cit in source_citations:
            citations_out.append(
                CitationOutput(
                    ref_number=cit.ref_number,
                    chunk_id=cit.chunk_id,
                    doc_id=cit.doc_id,
                    source_label=cit.source_label,
                    page=cit.page,
                    display_text=cit.display_text,
                )
            )

        return QueryToolOutput(
            query=input.query,
            answer=answer_text,
            citations=citations_out,
            abstained=abstained,
            candidate_count=len(result.candidates),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            generation_latency_ms=generation_latency_ms,
            elapsed_ms=result.elapsed_ms,
            error=result.error,
            run_id=result.run_id,
        )
    except Exception as exc:
        logger.exception("rag.query failed: %s", exc)
        return QueryToolOutput(
            query=input.query,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# rag.eval.run
# ---------------------------------------------------------------------------


@mcp.tool(
    name="rag.eval.run",
    description=(
        "Run a retrieval evaluation suite against the RAG corpus. "
        "Computes Recall@k, MRR, NDCG@k, and optional generation metrics."
    ),
)
def rag_eval_run(input: EvalRunToolInput) -> EvalRunToolOutput:
    """Run evaluation metrics over a labelled dataset.

    Args:
        input: Validated EvalRunToolInput payload.

    Returns:
        EvalRunToolOutput with per-metric results and elapsed time.

    Note:
        Full evaluation implementation is delivered in Phase 10.
        This stub returns a structured placeholder response.
    """
    logger.info("rag.eval.run called: dataset=%s metrics=%s", input.dataset_path, input.metrics)
    try:
        import json
        import time
        from pathlib import Path

        start = time.monotonic()

        dataset_path = Path(input.dataset_path)
        if not dataset_path.exists():
            return EvalRunToolOutput(
                dataset_path=input.dataset_path,
                error=f"Dataset file not found: {input.dataset_path}",
            )

        # Load dataset records
        records = []
        with dataset_path.open() as f:
            content = f.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                # JSONL
                records = [json.loads(line) for line in content.splitlines() if line.strip()]

        num_queries = len(records)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Phase 10 will compute real metrics — return placeholder zeros for now
        metric_results = [
            MetricResult(metric=m, value=0.0, num_samples=num_queries)
            for m in input.metrics
        ]

        output = EvalRunToolOutput(
            dataset_path=input.dataset_path,
            metrics=metric_results,
            num_queries=num_queries,
            elapsed_ms=elapsed_ms,
        )

        # Write report if requested
        if input.output_path:
            Path(input.output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(input.output_path).write_text(output.model_dump_json(indent=2))
            output.output_path = input.output_path

        return output

    except Exception as exc:
        logger.exception("rag.eval.run failed: %s", exc)
        return EvalRunToolOutput(
            dataset_path=input.dataset_path,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Entry point for running as a standalone MCP server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
