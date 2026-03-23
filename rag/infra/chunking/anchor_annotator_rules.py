"""Anchor annotator — detects structural anchors in TextBlocks using rule patterns."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.core.contracts.text_block import TextBlock

logger = logging.getLogger(__name__)


@dataclass
class AnchorAnnotation:
    """Structural anchor metadata for a single TextBlock.

    Attributes:
        anchor_type: Category of anchor — ``"heading"``, ``"section"``,
            ``"list"``, or ``"none"`` when no rule matched.
        anchor_level: Numeric depth within the anchor type (1 = top-level).
            0 when anchor_type is ``"none"``.
        rule_name: Name of the matching rule from anchors.yaml, or ``""``
            when no rule matched.
    """

    anchor_type: str = "none"
    anchor_level: int = 0
    rule_name: str = ""


@dataclass
class _Rule:
    """Internal compiled anchor rule."""

    name: str
    anchor_type: str
    pattern: re.Pattern
    level: int


def _load_rules(config_path: str | Path | None) -> list[_Rule]:
    """Load and compile anchor rules from YAML.

    Args:
        config_path: Path to anchors.yaml, or None for auto-detection.

    Returns:
        Ordered list of compiled rules.

    Raises:
        FileNotFoundError: If the config file cannot be found.
    """
    if config_path is None:
        candidate = Path(__file__).resolve()
        for parent in candidate.parents:
            yaml_path = parent / "configs" / "chunking" / "anchors.yaml"
            if yaml_path.exists():
                config_path = yaml_path
                break

    if config_path is None or not Path(config_path).exists():
        raise FileNotFoundError(
            "anchors.yaml not found. Set config_path explicitly."
        )

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    rules: list[_Rule] = []
    for entry in data.get("anchors", []):
        try:
            compiled = re.compile(entry["pattern"], re.MULTILINE)
            rules.append(
                _Rule(
                    name=entry["name"],
                    anchor_type=entry["type"],
                    pattern=compiled,
                    level=entry.get("level", 1),
                )
            )
        except re.error as exc:
            logger.warning("Invalid regex in rule '%s': %s", entry.get("name"), exc)

    return rules


class AnchorAnnotator:
    """Detects structural anchors in TextBlocks using configurable regex rules.

    Loads rule definitions from ``configs/chunking/anchors.yaml``. Each rule
    specifies a regex pattern; the first matching rule determines the anchor
    type and level for a block.

    Usage::

        annotator = AnchorAnnotator()
        annotations = annotator.annotate(text_blocks)
        # annotations[i] corresponds to text_blocks[i]

    Args:
        config_path: Path to anchors.yaml. If None, auto-detected from the
            project root.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._rules = _load_rules(config_path)

    def annotate_block(self, block: TextBlock) -> AnchorAnnotation:
        """Detect the anchor type and level for a single TextBlock.

        Evaluates rules in order; returns the first match. Returns
        ``AnchorAnnotation(anchor_type="none")`` if no rule matches.

        Args:
            block: TextBlock to evaluate.

        Returns:
            AnchorAnnotation describing the block's structural role.
        """
        # Use the block_type hint first — HEADING blocks are always anchors
        from rag.core.contracts.ir_block import BlockType

        if block.block_type == BlockType.HEADING:
            # Try to determine level from heading markers in text
            text = block.text.strip()
            level = 1
            if text.startswith("###"):
                level = 3
            elif text.startswith("##"):
                level = 2
            return AnchorAnnotation(anchor_type="heading", anchor_level=level, rule_name="block_type_heading")

        text = block.text.strip()
        for rule in self._rules:
            if rule.pattern.search(text):
                return AnchorAnnotation(
                    anchor_type=rule.anchor_type,
                    anchor_level=rule.level,
                    rule_name=rule.name,
                )

        return AnchorAnnotation()

    def annotate(self, blocks: list[TextBlock]) -> list[AnchorAnnotation]:
        """Annotate a list of TextBlocks with anchor metadata.

        Args:
            blocks: TextBlocks to annotate.

        Returns:
            List of AnchorAnnotations, one per input block, in the same order.
        """
        return [self.annotate_block(block) for block in blocks]
