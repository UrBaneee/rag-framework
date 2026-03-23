"""Parse quality gates — accepts or rejects a ParseReport against configured thresholds."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.core.contracts.parse_report import ParseReport

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result produced by QualityGateChecker.check().

    Attributes:
        passed: True if all enabled gates were satisfied.
        reasons: List of human-readable failure messages. Empty when passed=True.
    """

    passed: bool
    reasons: list[str] = field(default_factory=list)


class QualityGateChecker:
    """Evaluates a ParseReport against configurable quality thresholds.

    Loads gate definitions from ``configs/routers/quality_gates.yaml``.
    Each enabled gate checks one metric from the ParseReport and appends
    a failure reason when the threshold is violated.

    Usage::

        checker = QualityGateChecker()
        result = checker.check(document.parse_report)
        if not result.passed:
            print(result.reasons)

    Args:
        config_path: Path to quality_gates.yaml. If None, the default
            project config is used (auto-detected by walking up the tree).
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._gates = self._load_gates(config_path)

    def _load_gates(self, config_path: str | Path | None) -> dict:
        """Load gate configuration from YAML.

        Args:
            config_path: Path to quality_gates.yaml, or None for default.

        Returns:
            Dict of gate name → gate config dict.

        Raises:
            FileNotFoundError: If the config file cannot be found.
        """
        if config_path is None:
            candidate = Path(__file__).resolve()
            for parent in candidate.parents:
                yaml_path = parent / "configs" / "routers" / "quality_gates.yaml"
                if yaml_path.exists():
                    config_path = yaml_path
                    break

        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                "quality_gates.yaml not found. Set config_path explicitly."
            )

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data.get("gates", {})

    def check(self, report: ParseReport) -> GateResult:
        """Evaluate a ParseReport against all enabled quality gates.

        Args:
            report: ParseReport produced by a parser.

        Returns:
            GateResult with passed=True if all enabled gates are satisfied,
            or passed=False with a list of failure reasons.
        """
        reasons: list[str] = []

        gate = self._gates.get("min_char_count", {})
        if gate.get("enabled", False):
            threshold = gate["threshold"]
            if report.char_count < threshold:
                reasons.append(
                    f"char_count {report.char_count} < min {threshold}"
                )

        gate = self._gates.get("max_non_printable_ratio", {})
        if gate.get("enabled", False):
            threshold = gate["threshold"]
            if report.non_printable_ratio > threshold:
                reasons.append(
                    f"non_printable_ratio {report.non_printable_ratio:.3f} > max {threshold}"
                )

        gate = self._gates.get("max_repetition_score", {})
        if gate.get("enabled", False):
            threshold = gate["threshold"]
            if report.repetition_score > threshold:
                reasons.append(
                    f"repetition_score {report.repetition_score:.3f} > max {threshold}"
                )

        passed = len(reasons) == 0
        if not passed:
            logger.debug("Quality gate failed: %s", "; ".join(reasons))

        return GateResult(passed=passed, reasons=reasons)
