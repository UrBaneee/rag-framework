"""Tests for VoyageReranker — Task 6.2 (uses mocks, no real API calls)."""

from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.registry.plugin_registry import build_reranker
from rag.infra.rerank.voyage_rerank import VoyageReranker


# ---------------------------------------------------------------------------
# Helpers — mirror Voyage SDK return types
# ---------------------------------------------------------------------------

RerankingResult = namedtuple("RerankingResult", ["index", "document", "relevance_score"])


def _mock_reranking_object(scores: list[tuple[int, float]]) -> MagicMock:
    """Build a fake RerankingObject with the given (original_index, score) pairs."""
    obj = MagicMock()
    obj.results = [
        RerankingResult(index=idx, document=f"doc{idx}", relevance_score=score)
        for idx, score in scores
    ]
    obj.total_tokens = 42
    return obj


def _cand(chunk_id: str, text: str = "", rrf: float = 0.01) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        doc_id="doc1",
        display_text=text or f"text {chunk_id}",
        stable_text=text or f"stable {chunk_id}",
        rrf_score=rrf,
        final_score=rrf,
        retrieval_source=RetrievalSource.HYBRID,
    )


# ---------------------------------------------------------------------------
# Mock patch helper
# ---------------------------------------------------------------------------


def _make_reranker(mock_response) -> VoyageReranker:
    """Instantiate VoyageReranker with a mocked voyageai.Client."""
    with patch("voyageai.Client") as MockClient:
        instance = MockClient.return_value
        instance.rerank.return_value = mock_response
        reranker = VoyageReranker(api_key="test-key")
        reranker._client = instance
    return reranker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rerank_returns_reordered_candidates():
    """Voyage returns index 1 as best → chunk-b should be first."""
    candidates = [_cand("chunk-a"), _cand("chunk-b"), _cand("chunk-c")]
    # Voyage says: index 1 (chunk-b) score 0.9, index 0 (chunk-a) score 0.5
    mock_resp = _mock_reranking_object([(1, 0.9), (0, 0.5)])
    reranker = _make_reranker(mock_resp)
    reranker._client.rerank.return_value = mock_resp

    result = reranker.rerank("query", candidates, top_k=2)

    assert len(result) == 2
    assert result[0].chunk_id == "chunk-b"
    assert result[1].chunk_id == "chunk-a"


def test_rerank_sets_rerank_score():
    candidates = [_cand("a"), _cand("b")]
    mock_resp = _mock_reranking_object([(0, 0.95), (1, 0.60)])
    reranker = _make_reranker(mock_resp)
    reranker._client.rerank.return_value = mock_resp

    result = reranker.rerank("q", candidates, top_k=2)

    assert result[0].rerank_score == pytest.approx(0.95)
    assert result[1].rerank_score == pytest.approx(0.60)


def test_rerank_sets_final_score_equal_to_rerank_score():
    candidates = [_cand("a"), _cand("b")]
    mock_resp = _mock_reranking_object([(0, 0.7), (1, 0.3)])
    reranker = _make_reranker(mock_resp)
    reranker._client.rerank.return_value = mock_resp

    result = reranker.rerank("q", candidates, top_k=2)

    for c in result:
        assert c.final_score == c.rerank_score


def test_rerank_top_k_limits_output():
    candidates = [_cand(f"c{i}") for i in range(5)]
    # Voyage returns only top 3 (respects top_k passed to API)
    mock_resp = _mock_reranking_object([(2, 0.9), (0, 0.8), (4, 0.7)])
    reranker = _make_reranker(mock_resp)
    reranker._client.rerank.return_value = mock_resp

    result = reranker.rerank("q", candidates, top_k=3)

    assert len(result) == 3


def test_rerank_supports_top_8_output():
    """Acceptance criterion: supports top-8 output."""
    candidates = [_cand(f"c{i}") for i in range(10)]
    mock_resp = _mock_reranking_object([(i, 1.0 - i * 0.1) for i in range(8)])
    reranker = _make_reranker(mock_resp)
    reranker._client.rerank.return_value = mock_resp

    result = reranker.rerank("q", candidates, top_k=8)

    assert len(result) == 8


def test_rerank_empty_candidates_returns_empty():
    reranker = _make_reranker(MagicMock())
    result = reranker.rerank("q", [], top_k=5)
    assert result == []
    # API must not be called for empty input
    reranker._client.rerank.assert_not_called()


def test_rerank_passes_stable_text_to_api():
    """Voyage API must receive stable_text (not display_text)."""
    candidates = [
        _cand("a", text="stable content A"),
        _cand("b", text="stable content B"),
    ]
    mock_resp = _mock_reranking_object([(0, 0.8), (1, 0.5)])
    reranker = _make_reranker(mock_resp)
    reranker._client.rerank.return_value = mock_resp

    reranker.rerank("query", candidates, top_k=2)

    call_kwargs = reranker._client.rerank.call_args
    documents_sent = call_kwargs[1].get("documents") or call_kwargs[0][1]
    assert "stable content A" in documents_sent
    assert "stable content B" in documents_sent


def test_build_reranker_voyage_via_registry():
    """build_reranker with provider=voyage returns a VoyageReranker."""
    with patch("voyageai.Client"):
        reranker = build_reranker({
            "reranker": {"provider": "voyage", "api_key": "test-key"}
        })
    assert isinstance(reranker, VoyageReranker)
