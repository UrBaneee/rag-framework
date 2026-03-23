"""Tests for OpenAIEmbeddingProvider — all API calls are mocked."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.infra.embedding.openai_embedding import OpenAIEmbeddingProvider


def _make_response(vectors: list[list[float]], prompt_tokens: int = 10):
    """Build a fake openai embeddings response."""
    data = [
        SimpleNamespace(index=i, embedding=v) for i, v in enumerate(vectors)
    ]
    usage = SimpleNamespace(prompt_tokens=prompt_tokens)
    return SimpleNamespace(data=data, usage=usage)


@pytest.fixture()
def provider():
    """Provider with a mocked OpenAI client."""
    with patch("rag.infra.embedding.openai_embedding.OpenAI") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        p = OpenAIEmbeddingProvider(model="text-embedding-3-small", dim=4)
        p._client = mock_client
        yield p, mock_client


def test_embed_returns_correct_count(provider):
    p, client = provider
    client.embeddings.create.return_value = _make_response(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    )

    vectors = p.embed(["hello", "world"])

    assert len(vectors) == 2


def test_embed_returns_correct_dimensions(provider):
    p, client = provider
    client.embeddings.create.return_value = _make_response(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    )

    vectors = p.embed(["hello", "world"])

    assert all(len(v) == 4 for v in vectors)


def test_embed_raises_on_empty_input(provider):
    p, _ = provider
    with pytest.raises(ValueError, match="empty"):
        p.embed([])


def test_dim_property_matches_config(provider):
    p, _ = provider
    assert p.dim == 4


def test_embed_with_usage_returns_result(provider):
    p, client = provider
    client.embeddings.create.return_value = _make_response(
        [[1.0, 0.0, 0.0, 0.0]], prompt_tokens=5
    )

    result = p.embed_with_usage(["single text"])

    assert len(result.vectors) == 1
    assert result.model == "text-embedding-3-small"
    assert result.prompt_tokens == 5


def test_embed_uses_configured_model(provider):
    p, client = provider
    client.embeddings.create.return_value = _make_response([[0.0, 0.0, 0.0, 0.0]])

    p.embed(["test"])

    call_kwargs = client.embeddings.create.call_args
    assert call_kwargs.kwargs["model"] == "text-embedding-3-small"


def test_embed_batches_large_input():
    """Provider splits large input into batches of batch_size."""
    with patch("rag.infra.embedding.openai_embedding.OpenAI") as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        p = OpenAIEmbeddingProvider(model="text-embedding-3-small", dim=2, batch_size=2)
        p._client = mock_client

        # 5 texts, batch_size=2 → 3 API calls
        def side_effect(**kwargs):
            batch = kwargs["input"]
            return _make_response([[0.0, 0.0]] * len(batch))

        mock_client.embeddings.create.side_effect = side_effect

        vectors = p.embed(["a", "b", "c", "d", "e"])

        assert len(vectors) == 5
        assert mock_client.embeddings.create.call_count == 3
