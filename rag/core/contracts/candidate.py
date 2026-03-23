"""Candidate contract — a retrieved chunk with retrieval scores."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RetrievalSource(str, Enum):
    """Which retrieval system(s) surfaced this candidate."""

    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"


class Candidate(BaseModel):
    """A retrieved chunk with associated retrieval and reranking scores.

    Candidates flow through the retrieval pipeline:
      BM25 / Vector → RRF Fusion → (optional) Reranker → Context Packer

    Source attribution fields track which retrieval system(s) produced
    this candidate, supporting hybrid retrieval diagnostics.

    Attributes:
        chunk_id: Identifier of the underlying Chunk.
        doc_id: Parent document identifier.
        display_text: Human-readable text for citation and UI display.
        stable_text: Canonicalised text (used for reranking against query).
        bm25_score: Raw BM25 score, or None if not retrieved by BM25.
        vector_score: Cosine similarity score, or None if not retrieved by
            vector search.
        rrf_score: Reciprocal Rank Fusion score after fusion step.
        rerank_score: Cross-encoder or LLM rerank score, or None if
            reranking was skipped.
        final_score: The score used for final ordering (rerank_score if
            available, otherwise rrf_score).
        retrieval_source: Which retrieval path(s) produced this candidate.
        metadata: Forwarded chunk metadata (page, section, etc.).
    """

    chunk_id: str
    doc_id: str
    display_text: str
    stable_text: str
    bm25_score: Optional[float] = None
    vector_score: Optional[float] = None
    rrf_score: float = Field(default=0.0, ge=0.0)
    rerank_score: Optional[float] = None
    final_score: float = Field(default=0.0)
    retrieval_source: RetrievalSource = RetrievalSource.HYBRID
    metadata: dict = Field(default_factory=dict)

    @property
    def source_label(self) -> str:
        """Human-readable source attribution label.

        Returns one of:
        - ``"bm25_only"``   — retrieved by BM25 only
        - ``"vector_only"`` — retrieved by vector search only
        - ``"both"``        — retrieved by both BM25 and vector search

        Returns:
            Attribution string derived from which score fields are populated.
        """
        has_bm25 = self.bm25_score is not None
        has_vec = self.vector_score is not None
        if has_bm25 and has_vec:
            return "both"
        if has_bm25:
            return "bm25_only"
        if has_vec:
            return "vector_only"
        # Fallback: use retrieval_source enum value
        return self.retrieval_source.value
