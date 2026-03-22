"""Smoke tests — verify pytest infrastructure is working correctly."""

import pytest


@pytest.mark.unit
def test_smoke_passes() -> None:
    """Confirm that pytest can discover and run a basic test."""
    assert 1 + 1 == 2


@pytest.mark.unit
def test_sample_config_fixture(sample_config: dict) -> None:
    """Confirm that the sample_config fixture is available and well-formed."""
    assert "embedding" in sample_config
    assert "llm" in sample_config
    assert "retrieval" in sample_config
    assert "rerank" in sample_config
    assert "storage" in sample_config


@pytest.mark.unit
def test_tmp_path_fixture(tmp_path) -> None:
    """Confirm that pytest's built-in tmp_path fixture works."""
    test_file = tmp_path / "hello.txt"
    test_file.write_text("hello")
    assert test_file.read_text() == "hello"
