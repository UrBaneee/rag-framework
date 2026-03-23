"""Smoke tests — confirm storage and generation interfaces are importable and well-formed."""

import inspect

import pytest

from rag.core.interfaces.answer_composer import BaseAnswerComposer
from rag.core.interfaces.context_packer import BaseContextPacker
from rag.core.interfaces.doc_store import BaseDocStore
from rag.core.interfaces.llm_client import BaseLLMClient
from rag.core.interfaces.trace_store import BaseTraceStore


@pytest.mark.unit
def test_all_storage_gen_interfaces_importable():
    assert BaseDocStore is not None
    assert BaseTraceStore is not None
    assert BaseLLMClient is not None
    assert BaseContextPacker is not None
    assert BaseAnswerComposer is not None


@pytest.mark.unit
def test_base_doc_store_is_abstract():
    assert inspect.isabstract(BaseDocStore)
    expected = {"save_document", "get_document", "document_exists",
                "save_text_blocks", "get_text_blocks",
                "save_chunks", "get_chunks", "get_chunk_by_id", "delete_document"}
    assert expected.issubset(BaseDocStore.__abstractmethods__)


@pytest.mark.unit
def test_base_trace_store_is_abstract():
    assert inspect.isabstract(BaseTraceStore)
    expected = {"save_run", "save_answer_trace", "get_answer_trace", "list_runs"}
    assert expected.issubset(BaseTraceStore.__abstractmethods__)


@pytest.mark.unit
def test_base_llm_client_is_abstract():
    assert inspect.isabstract(BaseLLMClient)
    # complete() is a concrete convenience wrapper around generate(); not abstract
    assert "generate" in BaseLLMClient.__abstractmethods__
    assert "count_tokens" in BaseLLMClient.__abstractmethods__
    assert "model" in BaseLLMClient.__abstractmethods__


@pytest.mark.unit
def test_base_context_packer_is_abstract():
    assert inspect.isabstract(BaseContextPacker)
    assert "pack" in BaseContextPacker.__abstractmethods__


@pytest.mark.unit
def test_base_answer_composer_is_abstract():
    assert inspect.isabstract(BaseAnswerComposer)
    assert "compose" in BaseAnswerComposer.__abstractmethods__


@pytest.mark.unit
def test_cannot_instantiate_storage_gen_interfaces():
    with pytest.raises(TypeError):
        BaseDocStore()  # type: ignore
    with pytest.raises(TypeError):
        BaseTraceStore()  # type: ignore
    with pytest.raises(TypeError):
        BaseLLMClient()  # type: ignore
    with pytest.raises(TypeError):
        BaseContextPacker()  # type: ignore
    with pytest.raises(TypeError):
        BaseAnswerComposer()  # type: ignore
