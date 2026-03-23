"""Tests for embedding factory and base_embedding module."""

import pytest

from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.infra.embedding.base_embedding import EmbeddingResult
from rag.infra.embedding.embedding_factory import (
    create_embedding_provider,
    register_provider,
)


# ── Fake provider for testing ──────────────────────────────────────────────────

class FakeEmbeddingProvider(BaseEmbeddingProvider):
    """Minimal embedding provider that returns zero vectors."""

    def __init__(self, model: str = "fake-model", dim: int = 4, **kwargs) -> None:
        self._model = model
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            raise ValueError("texts must not be empty")
        return [[0.0] * self._dim for _ in texts]


# ── EmbeddingResult tests ──────────────────────────────────────────────────────

def test_embedding_result_len():
    result = EmbeddingResult(vectors=[[1.0, 2.0], [3.0, 4.0]], model="m")
    assert len(result) == 2


def test_embedding_result_stores_prompt_tokens():
    result = EmbeddingResult(vectors=[[0.0]], model="m", prompt_tokens=42)
    assert result.prompt_tokens == 42


# ── Factory tests ──────────────────────────────────────────────────────────────

def test_factory_returns_registered_provider(tmp_path, monkeypatch):
    """register_provider + create_embedding_provider returns correct instance."""
    # Register the fake provider under a test-only name
    register_provider(
        "fake",
        "tests.test_embedding_factory:FakeEmbeddingProvider",
    )

    config = {"embedding": {"provider": "fake", "model": "fake-model", "dim": 8}}
    provider = create_embedding_provider(config)

    assert isinstance(provider, BaseEmbeddingProvider)
    assert provider.dim == 8


def test_factory_provider_can_embed():
    register_provider(
        "fake",
        "tests.test_embedding_factory:FakeEmbeddingProvider",
    )

    config = {"embedding": {"provider": "fake", "model": "fake-model", "dim": 4}}
    provider = create_embedding_provider(config)

    vectors = provider.embed(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)


def test_factory_raises_for_unknown_provider():
    config = {"embedding": {"provider": "no_such_provider"}}
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        create_embedding_provider(config)


def test_factory_raises_for_missing_provider_key():
    with pytest.raises(KeyError):
        create_embedding_provider({"embedding": {}})


def test_fake_provider_raises_on_empty_texts():
    provider = FakeEmbeddingProvider()
    with pytest.raises(ValueError):
        provider.embed([])
