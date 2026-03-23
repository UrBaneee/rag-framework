"""MCP server wiring — builds pipeline objects from tool input schemas."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def build_ingest_pipeline(
    db_path: str,
    index_dir: str,
    token_budget: int = 512,
    embedding_provider: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    vector_dimension: int = 1536,
):
    """Instantiate and return a configured IngestPipeline.

    Args:
        db_path: SQLite database path.
        index_dir: Directory for BM25 and FAISS index files.
        token_budget: Chunk token budget.
        embedding_provider: Provider key (``"openai"``) or None to skip embedding.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.

    Returns:
        Configured ``IngestPipeline`` instance.
    """
    from rag.infra.stores.docstore_sqlite import SQLiteDocStore
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
    from rag.pipelines.ingest_pipeline import IngestPipeline

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    doc_store = SQLiteDocStore(db_path)
    trace_store = SQLiteTraceStore(db_path)

    embed_provider = None
    vec_index = None
    kw_index = None

    if embedding_provider == "openai":
        from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
        from rag.infra.indexes.index_manager import IndexManager

        embed_provider = OpenAIEmbeddingProvider(
            model=embedding_model,
            dimensions=vector_dimension,
        )
        mgr = IndexManager(index_dir=index_dir)
        vec_index = mgr.vector_index
        kw_index = mgr.keyword_index

    return IngestPipeline(
        doc_store=doc_store,
        trace_store=trace_store,
        token_budget=token_budget,
        embedding_provider=embed_provider,
        vector_index=vec_index,
        keyword_index=kw_index,
    )


def build_query_pipeline(
    db_path: str,
    index_dir: str,
    top_k: int = 10,
    context_top_k: int = 3,
    token_budget: int = 2048,
    embedding_provider: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small",
    vector_dimension: int = 1536,
    llm_model: str = "gpt-4o-mini",
    enable_generation: bool = True,
):
    """Instantiate and return a configured QueryPipeline.

    Args:
        db_path: SQLite database path.
        index_dir: Directory for BM25 and FAISS index files.
        top_k: Candidates to retrieve per index.
        context_top_k: Chunks to pack into the LLM context window.
        token_budget: Token budget for context packing.
        embedding_provider: Provider key (``"openai"``) or None.
        embedding_model: Model identifier for the embedding provider.
        vector_dimension: Expected vector dimensionality.
        llm_model: LLM model identifier for generation.
        enable_generation: Whether to wire the LLM composer.

    Returns:
        Configured ``QueryPipeline`` instance.
    """
    from rag.infra.stores.tracestore_sqlite import SQLiteTraceStore
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
            dimensions=vector_dimension,
        )
        vec_index = mgr.vector_index

    composer = None
    if enable_generation:
        from rag.infra.llm.openai_llm_client import OpenAILLMClient
        from rag.infra.generation.answer_composer_basic import BasicAnswerComposer
        llm = OpenAILLMClient(model=llm_model)
        composer = BasicAnswerComposer(
            llm_client=llm,
            top_k=context_top_k,
            token_budget=token_budget,
        )

    return QueryPipeline(
        keyword_index=mgr.keyword_index,
        trace_store=trace_store,
        vector_index=vec_index,
        embedding_provider=embed_provider,
        answer_composer=composer,
        top_k=top_k,
    )
