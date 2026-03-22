"""Smoke tests — confirm retrieval-side interfaces are importable and well-formed."""

import inspect

import pytest

from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.core.interfaces.keyword_index import BaseKeywordIndex
from rag.core.interfaces.reranker import BaseReranker
from rag.core.interfaces.vector_index import BaseVectorIndex


@pytest.mark.unit
def test_all_retrieval_interfaces_importable():
    assert BaseEmbeddingProvider is not None
    assert BaseVectorIndex is not None
    assert BaseKeywordIndex is not None
    assert BaseReranker is not None


@pytest.mark.unit
def test_base_embedding_provider_is_abstract():
    assert inspect.isabstract(BaseEmbeddingProvider)
    assert "embed" in BaseEmbeddingProvider.__abstractmethods__
    assert "dim" in BaseEmbeddingProvider.__abstractmethods__


@pytest.mark.unit
def test_base_vector_index_is_abstract():
    assert inspect.isabstract(BaseVectorIndex)
    assert "add" in BaseVectorIndex.__abstractmethods__
    assert "search" in BaseVectorIndex.__abstractmethods__
    assert "save" in BaseVectorIndex.__abstractmethods__
    assert "load" in BaseVectorIndex.__abstractmethods__


@pytest.mark.unit
def test_base_keyword_index_is_abstract():
    assert inspect.isabstract(BaseKeywordIndex)
    assert "add" in BaseKeywordIndex.__abstractmethods__
    assert "search" in BaseKeywordIndex.__abstractmethods__
    assert "save" in BaseKeywordIndex.__abstractmethods__
    assert "load" in BaseKeywordIndex.__abstractmethods__


@pytest.mark.unit
def test_base_reranker_is_abstract():
    assert inspect.isabstract(BaseReranker)
    assert "rerank" in BaseReranker.__abstractmethods__


@pytest.mark.unit
def test_cannot_instantiate_retrieval_interfaces():
    with pytest.raises(TypeError):
        BaseEmbeddingProvider()  # type: ignore
    with pytest.raises(TypeError):
        BaseVectorIndex()  # type: ignore
    with pytest.raises(TypeError):
        BaseKeywordIndex()  # type: ignore
    with pytest.raises(TypeError):
        BaseReranker()  # type: ignore
