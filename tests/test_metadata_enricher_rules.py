"""Tests for Task 12.2 — rules-based metadata enrichment.

Acceptance criteria:
- Chunks receive title/summary/tags using rules fallback
- Fields exist in metadata after enrichment
"""

from __future__ import annotations

import pytest

from rag.core.contracts.chunk import Chunk
from rag.infra.chunking.metadata_enricher_rules import EnrichmentResult, RulesMetadataEnricher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, chunk_id: str = "c1", metadata: dict | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        stable_text=text,
        display_text=text,
        chunk_signature=chunk_id,
        block_hashes=[chunk_id],
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Title extraction
# ---------------------------------------------------------------------------


class TestTitleExtraction:
    def test_markdown_heading_used_as_title(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk("## Introduction\n\nThis section explains the basics.")
        result = enricher.enrich(chunk)
        assert result.title == "Introduction"

    def test_first_sentence_used_when_no_heading(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk("The pipeline processes documents. It also handles embeddings.")
        result = enricher.enrich(chunk)
        assert "pipeline" in result.title.lower()

    def test_fallback_to_first_60_chars(self):
        enricher = RulesMetadataEnricher()
        # No heading, first sentence is too long (>80 chars)
        long_text = "A" * 90 + ". Short."
        chunk = _make_chunk(long_text)
        result = enricher.enrich(chunk)
        assert len(result.title) <= 60

    def test_title_written_to_metadata(self):
        enricher = RulesMetadataEnricher(annotate=True)
        chunk = _make_chunk("## Setup Guide\n\nFollow these steps.")
        enricher.enrich(chunk)
        assert "title" in chunk.metadata
        assert chunk.metadata["title"] == "Setup Guide"

    def test_title_not_overwritten_when_overwrite_false(self):
        enricher = RulesMetadataEnricher(annotate=True, overwrite=False)
        chunk = _make_chunk("## New Title\n\nContent.", metadata={"title": "Existing"})
        enricher.enrich(chunk)
        assert chunk.metadata["title"] == "Existing"

    def test_title_overwritten_when_overwrite_true(self):
        enricher = RulesMetadataEnricher(annotate=True, overwrite=True)
        chunk = _make_chunk("## New Title\n\nContent.", metadata={"title": "Old"})
        enricher.enrich(chunk)
        assert chunk.metadata["title"] == "New Title"


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------


class TestSummaryExtraction:
    def test_summary_uses_first_two_sentences(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk(
            "First sentence here. Second sentence here. Third sentence here."
        )
        result = enricher.enrich(chunk)
        assert "First sentence" in result.summary
        assert "Second sentence" in result.summary

    def test_summary_truncated_to_max_chars(self):
        enricher = RulesMetadataEnricher(max_summary_chars=20)
        chunk = _make_chunk("A very long sentence that goes on and on. Another sentence.")
        result = enricher.enrich(chunk)
        assert len(result.summary) <= 20

    def test_summary_written_to_metadata(self):
        enricher = RulesMetadataEnricher(annotate=True)
        chunk = _make_chunk("Pipeline overview. Handles documents efficiently.")
        enricher.enrich(chunk)
        assert "summary" in chunk.metadata
        assert isinstance(chunk.metadata["summary"], str)
        assert len(chunk.metadata["summary"]) > 0


# ---------------------------------------------------------------------------
# Tags extraction
# ---------------------------------------------------------------------------


class TestTagsExtraction:
    def test_has_code_tag_for_fenced_block(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk("Example code:\n```python\nprint('hello')\n```")
        result = enricher.enrich(chunk)
        assert "has_code" in result.tags

    def test_has_code_tag_for_inline_backtick(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk("Use `pip install` to install the package today.")
        result = enricher.enrich(chunk)
        assert "has_code" in result.tags

    def test_has_table_tag(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk("Results:\n| Col1 | Col2 |\n|---|---|\n| A | B |")
        result = enricher.enrich(chunk)
        assert "has_table" in result.tags

    def test_has_list_tag_for_bullet(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk("Steps:\n- First step\n- Second step\n- Third step")
        result = enricher.enrich(chunk)
        assert "has_list" in result.tags

    def test_has_list_tag_for_numbered(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk("Steps:\n1. Do this\n2. Do that\n3. Done")
        result = enricher.enrich(chunk)
        assert "has_list" in result.tags

    def test_high_pronoun_risk_tag(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk(
            "It is important that this works for them here.",
            metadata={"pronoun_risk": 0.6},
        )
        result = enricher.enrich(chunk)
        assert "high_pronoun_risk" in result.tags

    def test_no_high_pronoun_risk_below_threshold(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk(
            "The system handles requests efficiently.",
            metadata={"pronoun_risk": 0.1},
        )
        result = enricher.enrich(chunk)
        assert "high_pronoun_risk" not in result.tags

    def test_section_path_tags(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk(
            "Content of section.",
            metadata={"section_path": ["Architecture", "Retrieval"]},
        )
        result = enricher.enrich(chunk)
        assert "section:Architecture" in result.tags
        assert "section:Retrieval" in result.tags

    def test_clean_chunk_has_no_structural_tags(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk(
            "The pipeline processes documents using a modular architecture efficiently."
        )
        result = enricher.enrich(chunk)
        structural = {"has_code", "has_table", "has_list", "high_pronoun_risk"}
        assert not structural.intersection(result.tags)

    def test_tags_written_to_metadata(self):
        enricher = RulesMetadataEnricher(annotate=True)
        chunk = _make_chunk("Use `config.yaml` to configure the pipeline here.")
        enricher.enrich(chunk)
        assert "tags" in chunk.metadata
        assert isinstance(chunk.metadata["tags"], list)


# ---------------------------------------------------------------------------
# enrich_chunks() batch
# ---------------------------------------------------------------------------


class TestEnrichChunks:
    def test_returns_one_result_per_chunk(self):
        enricher = RulesMetadataEnricher()
        chunks = [_make_chunk(f"Sentence {i}. Another sentence.", f"c{i}") for i in range(4)]
        results = enricher.enrich_chunks(chunks)
        assert len(results) == 4

    def test_empty_input(self):
        enricher = RulesMetadataEnricher()
        assert enricher.enrich_chunks([]) == []

    def test_all_chunks_get_title_and_summary(self):
        enricher = RulesMetadataEnricher(annotate=True)
        chunks = [_make_chunk(f"Chunk number {i}. Contains relevant info.", f"c{i}") for i in range(3)]
        enricher.enrich_chunks(chunks)
        for chunk in chunks:
            assert "title" in chunk.metadata
            assert "summary" in chunk.metadata
            assert "tags" in chunk.metadata


# ---------------------------------------------------------------------------
# EnrichmentResult
# ---------------------------------------------------------------------------


class TestEnrichmentResult:
    def test_fields_set_populated(self):
        enricher = RulesMetadataEnricher(annotate=True)
        chunk = _make_chunk("The pipeline works. It handles queries well.")
        result = enricher.enrich(chunk)
        assert "title" in result.fields_set
        assert "summary" in result.fields_set
        assert "tags" in result.fields_set

    def test_fields_set_empty_when_not_annotating(self):
        enricher = RulesMetadataEnricher(annotate=False)
        chunk = _make_chunk("The pipeline works well here.")
        result = enricher.enrich(chunk)
        assert result.fields_set == []

    def test_result_has_chunk_id(self):
        enricher = RulesMetadataEnricher()
        chunk = _make_chunk("Content here.", "my-chunk")
        result = enricher.enrich(chunk)
        assert result.chunk_id == "my-chunk"
