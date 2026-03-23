"""Tests for parse quality gates."""

from pathlib import Path

import pytest
import yaml

from rag.core.contracts.parse_report import ParseReport
from rag.pipelines.parsing.quality_gates import GateResult, QualityGateChecker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(
    char_count: int = 500,
    non_printable_ratio: float = 0.0,
    repetition_score: float = 0.0,
    block_count: int = 5,
) -> ParseReport:
    return ParseReport(
        char_count=char_count,
        block_count=block_count,
        non_printable_ratio=non_printable_ratio,
        repetition_score=repetition_score,
        parser_used="test_parser",
        fallback_triggered=False,
    )


def _write_gates_config(tmp_path: Path, gates: dict) -> Path:
    p = tmp_path / "quality_gates.yaml"
    p.write_text(yaml.dump({"gates": gates}), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# GateResult dataclass
# ---------------------------------------------------------------------------


def test_gate_result_passed_no_reasons():
    result = GateResult(passed=True)
    assert result.passed is True
    assert result.reasons == []


def test_gate_result_failed_with_reasons():
    result = GateResult(passed=False, reasons=["char_count 10 < min 50"])
    assert result.passed is False
    assert len(result.reasons) == 1


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        QualityGateChecker(config_path=tmp_path / "nonexistent.yaml")


def test_default_config_loads():
    # Should find the project's quality_gates.yaml automatically
    checker = QualityGateChecker()
    assert checker._gates  # non-empty


# ---------------------------------------------------------------------------
# min_char_count gate
# ---------------------------------------------------------------------------


def test_char_count_passes_above_threshold(tmp_path):
    config = _write_gates_config(tmp_path, {
        "min_char_count": {"enabled": True, "threshold": 50},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(char_count=100))
    assert result.passed is True


def test_char_count_fails_below_threshold(tmp_path):
    config = _write_gates_config(tmp_path, {
        "min_char_count": {"enabled": True, "threshold": 50},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(char_count=10))
    assert result.passed is False
    assert any("char_count" in r for r in result.reasons)


def test_char_count_gate_disabled_skips_check(tmp_path):
    config = _write_gates_config(tmp_path, {
        "min_char_count": {"enabled": False, "threshold": 50},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(char_count=0))
    assert result.passed is True


# ---------------------------------------------------------------------------
# max_non_printable_ratio gate
# ---------------------------------------------------------------------------


def test_non_printable_passes_below_threshold(tmp_path):
    config = _write_gates_config(tmp_path, {
        "max_non_printable_ratio": {"enabled": True, "threshold": 0.05},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(non_printable_ratio=0.01))
    assert result.passed is True


def test_non_printable_fails_above_threshold(tmp_path):
    config = _write_gates_config(tmp_path, {
        "max_non_printable_ratio": {"enabled": True, "threshold": 0.05},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(non_printable_ratio=0.20))
    assert result.passed is False
    assert any("non_printable_ratio" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# max_repetition_score gate
# ---------------------------------------------------------------------------


def test_repetition_passes_below_threshold(tmp_path):
    config = _write_gates_config(tmp_path, {
        "max_repetition_score": {"enabled": True, "threshold": 0.4},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(repetition_score=0.1))
    assert result.passed is True


def test_repetition_fails_above_threshold(tmp_path):
    config = _write_gates_config(tmp_path, {
        "max_repetition_score": {"enabled": True, "threshold": 0.4},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(repetition_score=0.8))
    assert result.passed is False
    assert any("repetition_score" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Multiple gates together
# ---------------------------------------------------------------------------


def test_multiple_gates_all_pass(tmp_path):
    config = _write_gates_config(tmp_path, {
        "min_char_count": {"enabled": True, "threshold": 50},
        "max_non_printable_ratio": {"enabled": True, "threshold": 0.05},
        "max_repetition_score": {"enabled": True, "threshold": 0.4},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(char_count=200, non_printable_ratio=0.01, repetition_score=0.1))
    assert result.passed is True
    assert result.reasons == []


def test_multiple_gates_multiple_failures(tmp_path):
    config = _write_gates_config(tmp_path, {
        "min_char_count": {"enabled": True, "threshold": 50},
        "max_non_printable_ratio": {"enabled": True, "threshold": 0.05},
        "max_repetition_score": {"enabled": True, "threshold": 0.4},
    })
    checker = QualityGateChecker(config_path=config)
    result = checker.check(_make_report(char_count=5, non_printable_ratio=0.9, repetition_score=0.9))
    assert result.passed is False
    assert len(result.reasons) == 3


def test_only_failed_gates_reported(tmp_path):
    config = _write_gates_config(tmp_path, {
        "min_char_count": {"enabled": True, "threshold": 50},
        "max_non_printable_ratio": {"enabled": True, "threshold": 0.05},
    })
    checker = QualityGateChecker(config_path=config)
    # char_count fails, non_printable passes
    result = checker.check(_make_report(char_count=10, non_printable_ratio=0.01))
    assert result.passed is False
    assert len(result.reasons) == 1
    assert "char_count" in result.reasons[0]
