"""Tests for rag.core.utils.batching."""

import pytest

from rag.core.utils.batching import EmbedBatchAccumulator, iter_batches


# ── iter_batches ───────────────────────────────────────────────────────────────

def test_iter_batches_splits_evenly():
    batches = list(iter_batches([1, 2, 3, 4], batch_size=2))
    assert batches == [[1, 2], [3, 4]]


def test_iter_batches_handles_remainder():
    batches = list(iter_batches([1, 2, 3, 4, 5], batch_size=2))
    assert batches == [[1, 2], [3, 4], [5]]


def test_iter_batches_single_batch():
    batches = list(iter_batches([1, 2, 3], batch_size=10))
    assert batches == [[1, 2, 3]]


def test_iter_batches_empty_list():
    batches = list(iter_batches([], batch_size=4))
    assert batches == []


def test_iter_batches_batch_size_one():
    batches = list(iter_batches(["a", "b", "c"], batch_size=1))
    assert batches == [["a"], ["b"], ["c"]]


def test_iter_batches_raises_on_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        list(iter_batches([1, 2], batch_size=0))


def test_iter_batches_preserves_order():
    items = list(range(100))
    reconstructed = [x for batch in iter_batches(items, 7) for x in batch]
    assert reconstructed == items


# ── EmbedBatchAccumulator ──────────────────────────────────────────────────────

def test_accumulator_aggregates_vectors():
    acc = EmbedBatchAccumulator()
    acc.add([[1.0, 2.0], [3.0, 4.0]], prompt_tokens=10)
    acc.add([[5.0, 6.0]], prompt_tokens=5)
    assert acc.vectors == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_accumulator_sums_tokens():
    acc = EmbedBatchAccumulator()
    acc.add([[0.0]], prompt_tokens=12)
    acc.add([[0.0]], prompt_tokens=8)
    assert acc.total_tokens == 20


def test_accumulator_starts_empty():
    acc = EmbedBatchAccumulator()
    assert acc.vectors == []
    assert acc.total_tokens == 0


# ── Integration: 100 chunks via OpenAIEmbeddingProvider ───────────────────────

def test_openai_provider_batches_100_chunks():
    """100 texts with batch_size=10 must trigger exactly 10 API calls."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    def make_response(batch):
        data = [SimpleNamespace(index=i, embedding=[0.0] * 4) for i in range(len(batch))]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=len(batch)))

    with patch("rag.infra.embedding.openai_embedding.OpenAI") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        mock_client.embeddings.create.side_effect = lambda **kw: make_response(kw["input"])

        from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider
        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", dim=4, batch_size=10)
        provider._client = mock_client

        texts = [f"chunk {i}" for i in range(100)]
        result = provider.embed_with_usage(texts)

    assert len(result.vectors) == 100
    assert mock_client.embeddings.create.call_count == 10
    assert result.prompt_tokens == 100  # 10 tokens per batch × 10 batches
