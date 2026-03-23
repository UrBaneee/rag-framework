"""Tests for GroundedPromptBuilder — Task 7.4."""

import pytest

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.contracts.citation import Citation
from rag.infra.generation.context_packer_light import LightContextPacker, PackedContext
from rag.infra.generation.prompt_builder_grounded import BuiltPrompt, GroundedPromptBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(chunk_id: str, text: str, doc_id: str = "doc.pdf") -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id=doc_id,
        display_text=text,
        stable_text=text,
        rrf_score=1.0,
        final_score=1.0,
        retrieval_source=RetrievalSource.BM25,
        metadata={"source_label": f"{doc_id} — page 1"},
    )


def _pack(texts: list[str], top_k: int = 3) -> PackedContext:
    packer = LightContextPacker(top_k=top_k)
    cands = [_make_candidate(f"c{i}", t) for i, t in enumerate(texts)]
    return packer.pack(cands)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


def test_build_returns_built_prompt():
    builder = GroundedPromptBuilder()
    packed = _pack(["Some relevant context about RAG."])
    result = builder.build("What is RAG?", packed)
    assert isinstance(result, BuiltPrompt)


# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------


def test_system_contains_grounding_instruction():
    builder = GroundedPromptBuilder()
    packed = _pack(["Context chunk."])
    result = builder.build("Question?", packed)
    lower = result.system.lower()
    assert "grounded" in lower or "context" in lower or "citation" in lower


def test_system_contains_abstain_instruction():
    builder = GroundedPromptBuilder()
    packed = _pack(["Context chunk."])
    result = builder.build("Question?", packed)
    lower = result.system.lower()
    assert "insufficient" in lower or "don't have" in lower or "abstain" in lower


def test_custom_system_instructions():
    custom = "Custom system instruction for testing."
    builder = GroundedPromptBuilder(system_instructions=custom)
    packed = _pack(["Context chunk."])
    result = builder.build("Question?", packed)
    assert result.system == custom


# ---------------------------------------------------------------------------
# Context layout
# ---------------------------------------------------------------------------


def test_user_contains_context_passages():
    builder = GroundedPromptBuilder()
    packed = _pack(["RAG stands for Retrieval-Augmented Generation."])
    result = builder.build("What is RAG?", packed)
    assert "RAG stands for Retrieval-Augmented Generation." in result.user


def test_user_contains_all_packed_chunks():
    builder = GroundedPromptBuilder()
    packed = _pack(["Chunk alpha.", "Chunk beta.", "Chunk gamma."], top_k=3)
    result = builder.build("Question?", packed)
    assert "Chunk alpha." in result.user
    assert "Chunk beta." in result.user
    assert "Chunk gamma." in result.user


def test_user_numbers_passages():
    builder = GroundedPromptBuilder()
    packed = _pack(["First chunk.", "Second chunk."], top_k=2)
    result = builder.build("Question?", packed)
    assert "[1]" in result.user
    assert "[2]" in result.user


# ---------------------------------------------------------------------------
# Citation / source index
# ---------------------------------------------------------------------------


def test_user_contains_source_index():
    builder = GroundedPromptBuilder()
    packed = _pack(["Chunk from guide."])
    result = builder.build("Question?", packed)
    # Source label from metadata
    assert "doc.pdf" in result.user


def test_source_index_lists_all_citations():
    builder = GroundedPromptBuilder()
    cands = [
        _make_candidate("c0", "Alpha text.", doc_id="alpha.pdf"),
        _make_candidate("c1", "Beta text.", doc_id="beta.pdf"),
    ]
    packer = LightContextPacker(top_k=2)
    packed = packer.pack(cands)
    result = builder.build("Question?", packed)
    assert "alpha.pdf" in result.user
    assert "beta.pdf" in result.user


# ---------------------------------------------------------------------------
# Query embedding
# ---------------------------------------------------------------------------


def test_user_contains_query():
    builder = GroundedPromptBuilder()
    packed = _pack(["Some context."])
    result = builder.build("What is retrieval-augmented generation?", packed)
    assert "What is retrieval-augmented generation?" in result.user


def test_user_query_stripped():
    builder = GroundedPromptBuilder()
    packed = _pack(["Some context."])
    result = builder.build("  What is RAG?  ", packed)
    assert "What is RAG?" in result.user


# ---------------------------------------------------------------------------
# Abstain / empty context
# ---------------------------------------------------------------------------


def test_no_context_produces_abstain_hint():
    builder = GroundedPromptBuilder()
    empty_packed = PackedContext()  # no candidates, no context_text
    result = builder.build("What is RAG?", empty_packed)
    lower = result.user.lower()
    assert "no context" in lower or "abstain" in lower or "not available" in lower


# ---------------------------------------------------------------------------
# full_text property
# ---------------------------------------------------------------------------


def test_full_text_combines_system_and_user():
    builder = GroundedPromptBuilder()
    packed = _pack(["Context text."])
    result = builder.build("Question?", packed)
    assert result.system in result.full_text
    assert result.user in result.full_text
