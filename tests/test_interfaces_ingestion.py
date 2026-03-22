"""Smoke tests — confirm ingestion-side interfaces are importable and well-formed."""

import inspect

import pytest

from rag.core.interfaces.block_splitter import BaseBlockSplitter
from rag.core.interfaces.chunk_packer import BaseChunkPacker
from rag.core.interfaces.cleaner import BaseCleaner
from rag.core.interfaces.parser import BaseParser


@pytest.mark.unit
def test_all_ingestion_interfaces_importable():
    assert BaseParser is not None
    assert BaseCleaner is not None
    assert BaseBlockSplitter is not None
    assert BaseChunkPacker is not None


@pytest.mark.unit
def test_base_parser_is_abstract():
    assert inspect.isabstract(BaseParser)
    assert "parse" in BaseParser.__abstractmethods__
    assert "supports" in BaseParser.__abstractmethods__


@pytest.mark.unit
def test_base_cleaner_is_abstract():
    assert inspect.isabstract(BaseCleaner)
    assert "clean" in BaseCleaner.__abstractmethods__


@pytest.mark.unit
def test_base_block_splitter_is_abstract():
    assert inspect.isabstract(BaseBlockSplitter)
    assert "split" in BaseBlockSplitter.__abstractmethods__


@pytest.mark.unit
def test_base_chunk_packer_is_abstract():
    assert inspect.isabstract(BaseChunkPacker)
    assert "pack" in BaseChunkPacker.__abstractmethods__


@pytest.mark.unit
def test_cannot_instantiate_abstract_interfaces():
    with pytest.raises(TypeError):
        BaseParser()  # type: ignore
    with pytest.raises(TypeError):
        BaseCleaner()  # type: ignore
    with pytest.raises(TypeError):
        BaseBlockSplitter()  # type: ignore
    with pytest.raises(TypeError):
        BaseChunkPacker()  # type: ignore
