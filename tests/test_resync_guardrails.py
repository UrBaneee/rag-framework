"""Tests for Task 11.4 — resync threshold guardrails.

Acceptance criteria:
- Pipeline warns/records when changed ratio exceeds threshold
- configs/chunking/resync.yaml is loadable and contains valid thresholds
- Trace event is emitted with correct metadata
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from rag.core.utils.hashing import BlockDiffResult
from rag.pipelines.ingest_pipeline import IngestPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(warn=0.5, error=0.9) -> tuple[IngestPipeline, MagicMock, MagicMock]:
    """Build an IngestPipeline with mocked stores and given thresholds."""
    doc_store = MagicMock()
    trace_store = MagicMock()
    trace_store.save_run.return_value = "run-001"

    pipeline = IngestPipeline(doc_store=doc_store, trace_store=trace_store)
    # Override loaded thresholds directly for test isolation
    pipeline._warn_threshold = warn
    pipeline._error_threshold = error

    return pipeline, doc_store, trace_store


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------


class TestResyncConfigFile:
    def test_yaml_file_exists(self):
        config_path = Path("configs/chunking/resync.yaml")
        assert config_path.exists(), "configs/chunking/resync.yaml must exist"

    def test_yaml_contains_guardrails(self):
        import yaml

        config_path = Path("configs/chunking/resync.yaml")
        data = yaml.safe_load(config_path.read_text())
        assert "guardrails" in data
        assert "warn_threshold" in data["guardrails"]
        assert "error_threshold" in data["guardrails"]

    def test_yaml_thresholds_valid(self):
        import yaml

        config_path = Path("configs/chunking/resync.yaml")
        data = yaml.safe_load(config_path.read_text())
        warn = data["guardrails"]["warn_threshold"]
        error = data["guardrails"]["error_threshold"]
        assert 0.0 < warn < 1.0
        assert 0.0 < error <= 1.0
        assert warn < error

    def test_pipeline_loads_config(self):
        """Pipeline reads warn/error thresholds from resync.yaml at init."""
        doc_store = MagicMock()
        trace_store = MagicMock()
        trace_store.save_run.return_value = "run-001"
        config_path = Path("configs/chunking/resync.yaml")
        pipeline = IngestPipeline(
            doc_store=doc_store,
            trace_store=trace_store,
            resync_config_path=config_path,
        )
        assert 0.0 < pipeline._warn_threshold < 1.0
        assert 0.0 < pipeline._error_threshold <= 1.0


# ---------------------------------------------------------------------------
# Guardrail — no event below warn threshold
# ---------------------------------------------------------------------------


class TestGuardrailBelowThreshold:
    def test_no_trace_event_below_warn(self):
        pipeline, _, trace_store = _make_pipeline(warn=0.5, error=0.9)
        # 1 changed out of 10 = 10% < 50%
        diff = BlockDiffResult(unchanged=["a"] * 9, added=["x"], removed=[])
        pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        # save_run should NOT have been called for threshold events
        calls = [str(c) for c in trace_store.save_run.call_args_list]
        assert not any("resync_threshold_exceeded" in c for c in calls)

    def test_no_event_when_all_unchanged(self):
        pipeline, _, trace_store = _make_pipeline()
        # All blocks unchanged = no change at all
        diff = BlockDiffResult(unchanged=["a", "b", "c"], added=[], removed=[])
        pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        calls = [str(c) for c in trace_store.save_run.call_args_list]
        assert not any("resync_threshold_exceeded" in c for c in calls)


# ---------------------------------------------------------------------------
# Guardrail — warning threshold
# ---------------------------------------------------------------------------


class TestGuardrailWarning:
    def test_warning_log_emitted(self, caplog):
        pipeline, _, trace_store = _make_pipeline(warn=0.5, error=0.9)
        # 6 changed out of 10 = 60% >= 50%
        diff = BlockDiffResult(unchanged=["a"] * 4, added=["x"] * 6, removed=[])
        with caplog.at_level(logging.WARNING):
            pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        assert any("WARNING" in r.levelname or "guardrail" in r.message.lower()
                   for r in caplog.records)

    def test_warning_trace_event_emitted(self):
        pipeline, _, trace_store = _make_pipeline(warn=0.5, error=0.9)
        diff = BlockDiffResult(unchanged=["a"] * 4, added=["x"] * 6, removed=[])
        pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        trace_store.save_run.assert_called_once()
        call_kwargs = trace_store.save_run.call_args
        assert call_kwargs[1]["run_type"] == "resync_threshold_exceeded" or \
               call_kwargs[0][0] == "resync_threshold_exceeded" or \
               "resync_threshold_exceeded" in str(call_kwargs)

    def test_warning_metadata_contains_ratio(self):
        pipeline, _, trace_store = _make_pipeline(warn=0.5, error=0.9)
        diff = BlockDiffResult(unchanged=["a"] * 4, added=["x"] * 3, removed=["y"] * 3)
        pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        call_kwargs = trace_store.save_run.call_args
        metadata = call_kwargs[1].get("metadata") or call_kwargs[0][1]
        assert "changed_ratio" in metadata
        assert metadata["level"] == "warning"

    def test_warning_not_error_below_error_threshold(self):
        pipeline, _, trace_store = _make_pipeline(warn=0.5, error=0.9)
        # 60% >= warn but < error → should be "warning" not "error"
        diff = BlockDiffResult(unchanged=["a"] * 4, added=["x"] * 6, removed=[])
        pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        call_kwargs = trace_store.save_run.call_args
        metadata = call_kwargs[1].get("metadata") or call_kwargs[0][1]
        assert metadata["level"] == "warning"


# ---------------------------------------------------------------------------
# Guardrail — error threshold
# ---------------------------------------------------------------------------


class TestGuardrailError:
    def test_error_log_emitted(self, caplog):
        pipeline, _, trace_store = _make_pipeline(warn=0.5, error=0.9)
        # 9/10 changed = 90% >= 90%
        diff = BlockDiffResult(unchanged=["a"], added=["x"] * 9, removed=[])
        with caplog.at_level(logging.ERROR):
            pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        assert any(r.levelname == "ERROR" for r in caplog.records)

    def test_error_metadata_level(self):
        pipeline, _, trace_store = _make_pipeline(warn=0.5, error=0.9)
        diff = BlockDiffResult(unchanged=["a"], added=["x"] * 9, removed=[])
        pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        call_kwargs = trace_store.save_run.call_args
        metadata = call_kwargs[1].get("metadata") or call_kwargs[0][1]
        assert metadata["level"] == "error"

    def test_large_rewrite_triggers_error(self):
        """Simulate a large rewrite — all blocks replaced."""
        pipeline, _, trace_store = _make_pipeline(warn=0.5, error=0.9)
        diff = BlockDiffResult(unchanged=[], added=["x"] * 10, removed=["y"] * 10)
        pipeline._check_resync_guardrails(diff, "/tmp/doc.md", "run-001")
        call_kwargs = trace_store.save_run.call_args
        metadata = call_kwargs[1].get("metadata") or call_kwargs[0][1]
        assert metadata["level"] == "error"
        assert metadata["blocks_added"] == 10
        assert metadata["blocks_removed"] == 10


# ---------------------------------------------------------------------------
# IngestPipeline defaults
# ---------------------------------------------------------------------------


class TestPipelineGuardrailDefaults:
    def test_default_warn_threshold_set(self):
        doc_store = MagicMock()
        trace_store = MagicMock()
        trace_store.save_run.return_value = "run-001"
        pipeline = IngestPipeline(doc_store=doc_store, trace_store=trace_store)
        assert 0.0 < pipeline._warn_threshold <= 1.0

    def test_default_error_threshold_gt_warn(self):
        doc_store = MagicMock()
        trace_store = MagicMock()
        trace_store.save_run.return_value = "run-001"
        pipeline = IngestPipeline(doc_store=doc_store, trace_store=trace_store)
        assert pipeline._error_threshold >= pipeline._warn_threshold
