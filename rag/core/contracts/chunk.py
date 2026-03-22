"""Chunk contract — the unit stored in the vector and keyword indexes."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """An indexed chunk of text derived from one or more TextBlocks.

    A Chunk carries two text representations:
    - ``stable_text``: canonicalised text used for embedding and BM25 indexing.
      This must be deterministic so that re-ingestion of an unchanged document
      produces identical embeddings.
    - ``display_text``: human-readable text shown in the UI and used for
      citation construction. May include anchor prefixes injected by the
      anchor annotator.

    The ``chunk_signature`` is a SHA-256 over the ordered ``block_hashes``
    that compose this chunk, enabling efficient diff detection during
    incremental ingestion.

    Attributes:
        chunk_id: Stable identifier (SHA-256 of chunk_signature, or assigned
            by DocStore).
        doc_id: Parent document identifier.
        stable_text: Canonicalised text for embedding / BM25.
        display_text: Human-readable text for UI and citations.
        chunk_signature: SHA-256 over ordered block hashes.
        block_hashes: Ordered hashes of the TextBlocks that form this chunk.
        token_count: Approximate token count for context-packing budget.
        metadata: Arbitrary extra fields (page range, section path, etc.).
        embedding: Dense vector; populated after the embedding stage.
    """

    chunk_id: Optional[str] = None
    doc_id: str
    stable_text: str
    display_text: str
    chunk_signature: str
    block_hashes: list[str] = Field(default_factory=list)
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[list[float]] = None
