"""Unit tests for chunking and retrieval contracts: TextBlock, Chunk, Candidate."""

import pytest

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.contracts.chunk import Chunk
from rag.core.contracts.ir_block import BlockType
from rag.core.contracts.text_block import TextBlock


@pytest.mark.unit
class TestTextBlock:
    def test_text_block_required_fields(self):
        block = TextBlock(
            doc_id="doc-001",
            text="Some cleaned text.",
            block_hash="abc123",
            sequence=0,
        )
        assert block.doc_id == "doc-001"
        assert block.text == "Some cleaned text."
        assert block.block_hash == "abc123"
        assert block.sequence == 0
        assert block.block_type == BlockType.PARAGRAPH
        assert block.block_id is None
        assert block.page is None
        assert block.section_path == []

    def test_text_block_with_all_fields(self):
        block = TextBlock(
            block_id="blk-42",
            doc_id="doc-001",
            block_type=BlockType.HEADING,
            text="Introduction",
            block_hash="deadbeef",
            page=1,
            sequence=0,
            section_path=["Chapter 1"],
        )
        assert block.block_id == "blk-42"
        assert block.block_type == BlockType.HEADING
        assert block.page == 1
        assert block.section_path == ["Chapter 1"]

    def test_text_block_sequence_non_negative(self):
        with pytest.raises(Exception):
            TextBlock(doc_id="d", text="t", block_hash="h", sequence=-1)


@pytest.mark.unit
class TestChunk:
    def test_chunk_has_stable_and_display_text(self):
        chunk = Chunk(
            doc_id="doc-001",
            stable_text="deployment mode 3 cloud deployment all services are deployed on aws azure",
            display_text="Deployment Mode 3 — Cloud Deployment:\nAll services are deployed on AWS/Azure.",
            chunk_signature="sig-abc",
            block_hashes=["hash1", "hash2"],
        )
        assert chunk.stable_text != chunk.display_text
        assert chunk.chunk_signature == "sig-abc"
        assert chunk.block_hashes == ["hash1", "hash2"]
        assert chunk.token_count == 0
        assert chunk.embedding is None
        assert chunk.chunk_id is None

    def test_chunk_with_embedding(self):
        chunk = Chunk(
            doc_id="doc-001",
            stable_text="text",
            display_text="text",
            chunk_signature="sig",
            embedding=[0.1, 0.2, 0.3],
            token_count=3,
        )
        assert len(chunk.embedding) == 3
        assert chunk.token_count == 3

    def test_chunk_metadata(self):
        chunk = Chunk(
            doc_id="doc-001",
            stable_text="text",
            display_text="text",
            chunk_signature="sig",
            metadata={"page_start": 1, "page_end": 2},
        )
        assert chunk.metadata["page_start"] == 1


@pytest.mark.unit
class TestCandidate:
    def test_candidate_required_fields(self):
        candidate = Candidate(
            chunk_id="chk-001",
            doc_id="doc-001",
            display_text="Some display text.",
            stable_text="some display text",
        )
        assert candidate.chunk_id == "chk-001"
        assert candidate.rrf_score == 0.0
        assert candidate.bm25_score is None
        assert candidate.vector_score is None
        assert candidate.rerank_score is None
        assert candidate.retrieval_source == RetrievalSource.HYBRID

    def test_candidate_source_attribution_bm25(self):
        candidate = Candidate(
            chunk_id="c1",
            doc_id="d1",
            display_text="text",
            stable_text="text",
            bm25_score=12.5,
            retrieval_source=RetrievalSource.BM25,
            rrf_score=0.016,
            final_score=0.016,
        )
        assert candidate.retrieval_source == RetrievalSource.BM25
        assert candidate.bm25_score == 12.5
        assert candidate.vector_score is None

    def test_candidate_source_attribution_vector(self):
        candidate = Candidate(
            chunk_id="c2",
            doc_id="d1",
            display_text="text",
            stable_text="text",
            vector_score=0.92,
            retrieval_source=RetrievalSource.VECTOR,
            rrf_score=0.015,
            final_score=0.015,
        )
        assert candidate.retrieval_source == RetrievalSource.VECTOR
        assert candidate.vector_score == 0.92

    def test_candidate_with_rerank_score(self):
        candidate = Candidate(
            chunk_id="c3",
            doc_id="d1",
            display_text="text",
            stable_text="text",
            rrf_score=0.014,
            rerank_score=0.87,
            final_score=0.87,
        )
        assert candidate.rerank_score == 0.87
        assert candidate.final_score == 0.87
