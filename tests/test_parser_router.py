"""Tests for the parser router (orchestrator + plans)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.infra.loading.local_file_loader import RawArtifact
from rag.infra.sniffing.composite_sniffer import SniffResult
from rag.pipelines.parsing.orchestrator import ParserOrchestrator, _unsupported_document
from rag.pipelines.parsing.plans import ParsePlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_artifact(source_path: str = "/tmp/test.pdf") -> RawArtifact:
    return RawArtifact(
        source_path=source_path,
        raw_bytes=b"%PDF-1.4 fake content",
        metadata={"extension": "pdf"},
    )


def _make_document(parser_name: str = "mock_parser") -> Document:
    block = IRBlock(block_type=BlockType.PARAGRAPH, text="Hello from parser")
    report = ParseReport(
        char_count=17,
        block_count=1,
        non_printable_ratio=0.0,
        repetition_score=0.0,
        parser_used=parser_name,
        fallback_triggered=False,
    )
    return Document(
        doc_id="abc123",
        source_path="/tmp/test.pdf",
        mime_type="application/pdf",
        metadata={},
        blocks=[block],
        parse_report=report,
    )


def _make_sniff_result(detected_type: str = "pdf") -> SniffResult:
    return SniffResult(
        detected_type=detected_type,
        mime_type="application/pdf",
        confidence=1.0,
        strategy="magic",
    )


def _make_yaml_config(tmp_path: Path, routes: dict) -> Path:
    """Write a minimal parser_candidates.yaml to tmp_path and return its path."""
    import yaml

    config_path = tmp_path / "parser_candidates.yaml"
    config_path.write_text(yaml.dump({"routes": routes}), encoding="utf-8")
    return config_path


# ---------------------------------------------------------------------------
# ParsePlan tests
# ---------------------------------------------------------------------------


def test_parseplan_defaults():
    plan = ParsePlan(detected_type="pdf", mime_type="application/pdf")
    assert plan.candidates == []
    assert plan.unsupported is False


def test_parseplan_unsupported_flag():
    plan = ParsePlan(
        detected_type="unsupported",
        mime_type="application/octet-stream",
        candidates=[],
        unsupported=True,
    )
    assert plan.unsupported is True


# ---------------------------------------------------------------------------
# _unsupported_document helper
# ---------------------------------------------------------------------------


def test_unsupported_document_structure():
    artifact = _make_artifact()
    doc = _unsupported_document(artifact, "weird_format")
    assert doc.metadata["unsupported_format"] is True
    assert doc.metadata["detected_type"] == "weird_format"
    assert len(doc.blocks) == 1
    assert "unsupported_format" in doc.blocks[0].text
    assert doc.parse_report is not None


# ---------------------------------------------------------------------------
# ParserOrchestrator — route()
# ---------------------------------------------------------------------------


def test_route_known_type_returns_candidates(tmp_path):
    config = _make_yaml_config(tmp_path, {"pdf": {"candidates": ["pymupdf"]}})
    orch = ParserOrchestrator({}, router_config_path=config)
    sniff = _make_sniff_result("pdf")
    plan = orch.route(sniff)
    assert plan.detected_type == "pdf"
    assert plan.candidates == ["pymupdf"]
    assert plan.unsupported is False


def test_route_unknown_type_marks_unsupported(tmp_path):
    config = _make_yaml_config(tmp_path, {})
    orch = ParserOrchestrator({}, router_config_path=config)
    sniff = _make_sniff_result("unknown")
    plan = orch.route(sniff)
    assert plan.unsupported is True
    assert plan.candidates == []


def test_route_type_not_in_config_marks_unsupported(tmp_path):
    config = _make_yaml_config(tmp_path, {"pdf": {"candidates": ["pymupdf"]}})
    orch = ParserOrchestrator({}, router_config_path=config)
    sniff = _make_sniff_result("csv")
    plan = orch.route(sniff)
    assert plan.unsupported is True


# ---------------------------------------------------------------------------
# ParserOrchestrator — parse()
# ---------------------------------------------------------------------------


def test_parse_first_parser_succeeds(tmp_path):
    """Primary parser succeeds — no fallback triggered."""
    config = _make_yaml_config(tmp_path, {"pdf": {"candidates": ["parser_a", "parser_b"]}})

    doc_a = _make_document("parser_a")
    parser_a = MagicMock()
    parser_a.parse.return_value = doc_a

    parser_b = MagicMock()

    orch = ParserOrchestrator(
        {"parser_a": parser_a, "parser_b": parser_b},
        router_config_path=config,
    )
    plan = ParsePlan(detected_type="pdf", mime_type="application/pdf", candidates=["parser_a", "parser_b"])
    artifact = _make_artifact()

    result = orch.parse(artifact, plan)

    assert result.doc_id == doc_a.doc_id
    parser_a.parse.assert_called_once_with(artifact.source_path)
    parser_b.parse.assert_not_called()
    assert result.parse_report.fallback_triggered is False


def test_parse_fallback_to_second_parser(tmp_path):
    """First parser fails, second succeeds — fallback_triggered set to True."""
    config = _make_yaml_config(tmp_path, {"pdf": {"candidates": ["parser_a", "parser_b"]}})

    doc_b = _make_document("parser_b")
    parser_a = MagicMock()
    parser_a.parse.side_effect = RuntimeError("parser_a exploded")

    parser_b = MagicMock()
    parser_b.parse.return_value = doc_b

    orch = ParserOrchestrator(
        {"parser_a": parser_a, "parser_b": parser_b},
        router_config_path=config,
    )
    plan = ParsePlan(detected_type="pdf", mime_type="application/pdf", candidates=["parser_a", "parser_b"])
    artifact = _make_artifact()

    result = orch.parse(artifact, plan)

    assert result.doc_id == doc_b.doc_id
    assert result.parse_report.fallback_triggered is True
    parser_a.parse.assert_called_once()
    parser_b.parse.assert_called_once()


def test_parse_all_parsers_fail_returns_unsupported(tmp_path):
    """All parsers fail — returns unsupported Document."""
    config = _make_yaml_config(tmp_path, {"pdf": {"candidates": ["parser_a"]}})

    parser_a = MagicMock()
    parser_a.parse.side_effect = RuntimeError("dead")

    orch = ParserOrchestrator({"parser_a": parser_a}, router_config_path=config)
    plan = ParsePlan(detected_type="pdf", mime_type="application/pdf", candidates=["parser_a"])
    artifact = _make_artifact()

    result = orch.parse(artifact, plan)

    assert result.metadata["unsupported_format"] is True


def test_parse_unsupported_plan_skips_parsing(tmp_path):
    """Unsupported plan short-circuits without calling any parser."""
    config = _make_yaml_config(tmp_path, {})
    parser_a = MagicMock()
    orch = ParserOrchestrator({"parser_a": parser_a}, router_config_path=config)
    plan = ParsePlan(detected_type="unsupported", mime_type="", candidates=[], unsupported=True)
    artifact = _make_artifact()

    result = orch.parse(artifact, plan)

    assert result.metadata["unsupported_format"] is True
    parser_a.parse.assert_not_called()


def test_parse_unregistered_parser_skipped(tmp_path):
    """If a candidate name is not in the registry, it is skipped gracefully."""
    config = _make_yaml_config(tmp_path, {"pdf": {"candidates": ["ghost_parser", "real_parser"]}})

    doc = _make_document("real_parser")
    real_parser = MagicMock()
    real_parser.parse.return_value = doc

    orch = ParserOrchestrator({"real_parser": real_parser}, router_config_path=config)
    plan = ParsePlan(
        detected_type="pdf",
        mime_type="application/pdf",
        candidates=["ghost_parser", "real_parser"],
    )
    artifact = _make_artifact()

    result = orch.parse(artifact, plan)

    assert result.doc_id == doc.doc_id
    real_parser.parse.assert_called_once()


# ---------------------------------------------------------------------------
# Config loading — missing file raises
# ---------------------------------------------------------------------------


def test_load_routes_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ParserOrchestrator({}, router_config_path="/nonexistent/path/parser_candidates.yaml")
