"""Unit tests for the config loader."""

import pytest

from rag.core.registry.config_loader import _deep_merge, load_config


@pytest.mark.unit
class TestDeepMerge:
    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"retrieval": {"bm25_top_k": 20, "vector_top_k": 20}}
        override = {"retrieval": {"bm25_top_k": 10}}
        result = _deep_merge(base, override)
        assert result["retrieval"]["bm25_top_k"] == 10
        assert result["retrieval"]["vector_top_k"] == 20  # preserved

    def test_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert "y" not in base["a"]

    def test_deep_nested_merge(self):
        base = {"level1": {"level2": {"keep": True, "replace": "old"}}}
        override = {"level1": {"level2": {"replace": "new", "add": "added"}}}
        result = _deep_merge(base, override)
        assert result["level1"]["level2"]["keep"] is True
        assert result["level1"]["level2"]["replace"] == "new"
        assert result["level1"]["level2"]["add"] == "added"


@pytest.mark.unit
class TestLoadConfig:
    def test_load_base_config(self):
        config = load_config()
        # Retrieval defaults from settings.yaml
        assert config["retrieval"]["bm25_top_k"] == 20
        assert config["retrieval"]["vector_top_k"] == 20
        assert config["retrieval"]["rerank_top_k"] == 8
        assert config["retrieval"]["rrf_k"] == 60
        # Generation defaults
        assert config["generation"]["context_top_k"] == 3
        assert config["generation"]["token_budget"] == 3000
        assert config["generation"]["abstain_if_empty"] is True
        # Reranking defaults
        assert config["reranking"]["enabled"] is True

    def test_load_local_fast_profile(self):
        config = load_config(profile="local_fast")
        # Profile overrides
        assert config["retrieval"]["bm25_top_k"] == 10
        assert config["retrieval"]["vector_top_k"] == 10
        assert config["retrieval"]["fusion_pool_size"] == 20
        assert config["reranking"]["enabled"] is False
        assert config["generation"]["context_top_k"] == 2
        assert config["generation"]["token_budget"] == 1500
        # Base values not in profile are preserved
        assert config["retrieval"]["rrf_k"] == 60
        assert config["generation"]["abstain_if_empty"] is True

    def test_load_local_quality_profile(self):
        config = load_config(profile="local_quality")
        # Profile overrides
        assert config["reranking"]["enabled"] is True
        assert config["reranking"]["provider"] == "voyage"
        assert config["generation"]["llm_model"] == "gpt-4o"
        assert config["generation"]["context_top_k"] == 3
        # Base values preserved
        assert config["retrieval"]["rrf_k"] == 60

    def test_unknown_profile_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config(profile="nonexistent_profile")

    def test_base_config_not_mutated_by_profile(self):
        base = load_config()
        load_config(profile="local_fast")
        base_again = load_config()
        assert base_again["retrieval"]["bm25_top_k"] == base["retrieval"]["bm25_top_k"]
