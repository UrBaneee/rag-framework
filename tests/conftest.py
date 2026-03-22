"""Shared pytest fixtures and configuration for the RAG framework test suite."""

import pytest


# ── Marker registration ────────────────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers to avoid PytestUnknownMarkWarning."""
    config.addinivalue_line("markers", "unit: fast, isolated unit tests")
    config.addinivalue_line("markers", "integration: tests that touch real infrastructure (db, disk)")
    config.addinivalue_line("markers", "e2e: end-to-end pipeline tests")


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def sample_config() -> dict:
    """Return a minimal in-memory config dict suitable for unit tests.

    Returns:
        Dictionary with default settings for all major subsystems.
    """
    return {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dim": 1536,
        },
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        "retrieval": {
            "top_k": 5,
            "bm25_top_k": 20,
            "vector_top_k": 20,
        },
        "rerank": {
            "enabled": True,
            "provider": "voyage",
        },
        "storage": {
            "data_dir": "/tmp/rag_test",
        },
    }
