"""Trace contracts — pipeline observability for query and ingestion runs."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class PipelineStep(BaseModel):
    """A single recorded step in a pipeline execution trace.

    Attributes:
        step_name: Identifier for the pipeline stage, e.g. "bm25_retrieval".
        input_summary: Brief description of inputs (not full data).
        output_summary: Brief description of outputs.
        latency_ms: Wall-clock time for this step in milliseconds.
        metadata: Arbitrary extra diagnostic fields.
    """

    step_name: str
    input_summary: str = ""
    output_summary: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerTrace(BaseModel):
    """Full observability record for a single query pipeline run.

    Captures token usage, latency, model metadata, and per-step traces
    so that developers can inspect exactly how an answer was produced.

    Attributes:
        query: The original user query.
        prompt_tokens: Number of tokens in the LLM prompt.
        completion_tokens: Number of tokens in the LLM response.
        total_tokens: Sum of prompt and completion tokens.
        total_latency_ms: Wall-clock time for the entire pipeline in ms.
        model: LLM model identifier used for generation.
        rerank_provider: Reranker used, or None if reranking was skipped.
        candidates_before_rerank: Number of candidates entering the reranker.
        candidates_after_rerank: Number of candidates after reranking.
        context_chunks_used: Number of chunks packed into the prompt context.
        steps: Ordered list of per-stage PipelineStep records.
        run_id: Optional identifier linking this trace to a TraceStore run.
    """

    query: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    model: str = ""
    rerank_provider: Optional[str] = None
    candidates_before_rerank: int = Field(default=0, ge=0)
    candidates_after_rerank: int = Field(default=0, ge=0)
    context_chunks_used: int = Field(default=0, ge=0)
    steps: list[PipelineStep] = Field(default_factory=list)
    run_id: Optional[str] = None
