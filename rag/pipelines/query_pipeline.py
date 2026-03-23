"""Query pipeline — retrieval, fusion, optional reranking, and answer generation.

This module is built incrementally across Phase 5 tasks:
  5.1  — attribute_candidates() source attribution merger
  5.2  — RRF fusion (rrf_fusion.py)
  5.3  — Full QueryPipeline (BM25 + vector → fusion → citations)
  5.4+ — Reranking, context packing, LLM generation
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.contracts.citation import Citation
from rag.core.interfaces.doc_store import BaseDocStore
from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.core.interfaces.keyword_index import BaseKeywordIndex
from rag.core.interfaces.trace_store import BaseTraceStore
from rag.core.interfaces.vector_index import BaseVectorIndex
from rag.infra.indexes.rrf_fusion import RRFFusion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    """Output of a single query pipeline run.

    Attributes:
        query: The original query string.
        candidates: RRF-fused and ranked candidates (top_k).
        citations: Citation objects derived from the top candidates.
        run_id: TraceStore run identifier for observability.
        elapsed_ms: Total wall-clock time in milliseconds.
        error: Error message if the query failed, or None on success.
    """

    query: str
    candidates: list[Candidate] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    run_id: str = ""
    elapsed_ms: float = 0.0
    error: Optional[str] = None


def attribute_candidates(
    bm25_results: list[Candidate],
    vector_results: list[Candidate],
) -> list[Candidate]:
    """Merge BM25 and vector candidate lists with source attribution.

    Candidates appearing in only one list are marked ``bm25_only`` or
    ``vector_only``. Candidates appearing in both lists are merged into a
    single ``Candidate`` carrying both scores and marked ``both``.

    The merged list preserves all unique chunk_ids from both inputs.
    Order within each source list is not guaranteed to be preserved — callers
    should apply RRF fusion or another ranking step afterwards.

    Args:
        bm25_results: Candidates returned by the BM25 keyword index.
        vector_results: Candidates returned by the FAISS vector index.

    Returns:
        Merged list of Candidates with ``retrieval_source`` and score fields
        set to reflect which system(s) surfaced each candidate.
    """
    # Index BM25 results by chunk_id
    bm25_by_id: dict[str, Candidate] = {c.chunk_id: c for c in bm25_results}
    # Index vector results by chunk_id
    vec_by_id: dict[str, Candidate] = {c.chunk_id: c for c in vector_results}

    merged: dict[str, Candidate] = {}

    # BM25-only or both
    for chunk_id, bm25_cand in bm25_by_id.items():
        if chunk_id in vec_by_id:
            vec_cand = vec_by_id[chunk_id]
            merged[chunk_id] = bm25_cand.model_copy(
                update={
                    "vector_score": vec_cand.vector_score,
                    "retrieval_source": RetrievalSource.HYBRID,
                }
            )
        else:
            merged[chunk_id] = bm25_cand.model_copy(
                update={"retrieval_source": RetrievalSource.BM25}
            )

    # Vector-only (not already in merged)
    for chunk_id, vec_cand in vec_by_id.items():
        if chunk_id not in merged:
            merged[chunk_id] = vec_cand.model_copy(
                update={"retrieval_source": RetrievalSource.VECTOR}
            )

    result = list(merged.values())
    logger.debug(
        "attribute_candidates: %d bm25, %d vector → %d merged "
        "(%d bm25_only, %d vector_only, %d both)",
        len(bm25_results),
        len(vector_results),
        len(result),
        sum(1 for c in result if c.source_label == "bm25_only"),
        sum(1 for c in result if c.source_label == "vector_only"),
        sum(1 for c in result if c.source_label == "both"),
    )
    return result


# ---------------------------------------------------------------------------
# Citation builder
# ---------------------------------------------------------------------------


def _build_citations(candidates: list[Candidate]) -> list[Citation]:
    """Build a Citation list from the top-ranked candidates.

    Args:
        candidates: RRF-fused candidates in ranked order.

    Returns:
        Citations with 1-based ref_number, one per candidate.
    """
    citations: list[Citation] = []
    for i, cand in enumerate(candidates, start=1):
        page = cand.metadata.get("start_page") or cand.metadata.get("page")
        source_path = cand.metadata.get("source_path", cand.doc_id)
        source_filename = Path(source_path).name if source_path else cand.doc_id
        source_label = source_filename
        if page:
            source_label = f"{source_filename} — page {page}"
        citations.append(
            Citation(
                ref_number=i,
                chunk_id=cand.chunk_id,
                doc_id=cand.doc_id,
                source_label=source_label,
                page=page,
                display_text=cand.display_text[:200],
            )
        )
    return citations


# ---------------------------------------------------------------------------
# QueryPipeline
# ---------------------------------------------------------------------------


class QueryPipeline:
    """Hybrid retrieval query pipeline — BM25 + vector → RRF → citations.

    Runs BM25 and (optionally) vector search, merges results with source
    attribution, fuses using RRF, and returns ranked candidates with
    citations. Does not perform LLM generation (added in a later phase).

    Query run metadata is recorded in the TraceStore for observability.

    Usage::

        pipeline = QueryPipeline(
            keyword_index=bm25,
            vector_index=faiss,
            embedding_provider=provider,
            trace_store=trace_store,
        )
        result = pipeline.query("What is retrieval augmented generation?")

    Args:
        keyword_index: BM25 or other keyword index.
        trace_store: TraceStore for recording query runs.
        vector_index: FAISS or other vector index. Optional — if omitted,
            only BM25 retrieval is performed.
        embedding_provider: Used to embed the query for vector search.
            Required when ``vector_index`` is provided.
        top_k: Number of candidates to retrieve per index. Defaults to 10.
        rrf_k: Smoothing constant for RRF fusion. Defaults to 60.
    """

    def __init__(
        self,
        keyword_index: BaseKeywordIndex,
        trace_store: BaseTraceStore,
        vector_index: Optional[BaseVectorIndex] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self._keyword_index = keyword_index
        self._vector_index = vector_index
        self._embedding_provider = embedding_provider
        self._trace_store = trace_store
        self._top_k = top_k
        self._fusion = RRFFusion(k=rrf_k)

    def query(self, query: str) -> QueryResult:
        """Run the full hybrid retrieval pipeline for a query string.

        Args:
            query: Natural-language query from the user.

        Returns:
            QueryResult with ranked candidates, citations, and run metadata.
        """
        start = time.monotonic()
        run_id = self._trace_store.save_run(
            run_type="query",
            metadata={"query": query, "top_k": self._top_k},
        )

        try:
            result = self._run(query, run_id)
        except Exception as exc:
            logger.exception("Query failed for '%s': %s", query, exc)
            result = QueryResult(query=query, run_id=run_id, error=str(exc))

        result.elapsed_ms = (time.monotonic() - start) * 1000
        self._trace_store.save_run(
            run_type="query_complete",
            metadata={
                "query": query,
                "run_id": run_id,
                "candidate_count": len(result.candidates),
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
            },
        )
        return result

    def _run(self, query: str, run_id: str) -> QueryResult:
        """Internal query execution.

        Args:
            query: Raw query string.
            run_id: TraceStore run identifier.

        Returns:
            Populated QueryResult on success.
        """
        # 1. BM25 retrieval
        bm25_results = self._keyword_index.search(query, top_k=self._top_k)
        logger.debug("BM25 returned %d candidates.", len(bm25_results))

        # 2. Vector retrieval (optional)
        vector_results: list[Candidate] = []
        if self._vector_index is not None and self._embedding_provider is not None:
            query_vec = self._embedding_provider.embed([query])[0]
            vector_results = self._vector_index.search(query_vec, top_k=self._top_k)
            logger.debug("Vector search returned %d candidates.", len(vector_results))

        # 3. Source attribution merge
        attributed = attribute_candidates(bm25_results, vector_results)

        # 4. RRF fusion — feed original ranked lists for correct rank positions
        ranked_lists = [l for l in [bm25_results, vector_results] if l]
        if ranked_lists:
            fused = self._fusion.fuse(ranked_lists)[: self._top_k]
        else:
            fused = []

        # 5. Build citations
        citations = _build_citations(fused)

        logger.info(
            "Query '%s': %d candidates after fusion (run=%s)", query, len(fused), run_id
        )

        return QueryResult(
            query=query,
            candidates=fused,
            citations=citations,
            run_id=run_id,
        )
