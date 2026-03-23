"""Tests for the AnchorAnnotator."""

from pathlib import Path

import pytest
import yaml

from rag.core.contracts.ir_block import BlockType
from rag.core.contracts.text_block import TextBlock
from rag.infra.chunking.anchor_annotator_rules import AnchorAnnotation, AnchorAnnotator

DOC_ID = "doc_001"


def _tb(text: str, block_type: BlockType = BlockType.PARAGRAPH, sequence: int = 0) -> TextBlock:
    return TextBlock(
        doc_id=DOC_ID,
        block_type=block_type,
        text=text,
        block_hash="abc" * 21 + "d",  # 64-char placeholder
        sequence=sequence,
    )


def _write_config(tmp_path: Path, anchors: list[dict]) -> Path:
    p = tmp_path / "anchors.yaml"
    p.write_text(yaml.dump({"anchors": anchors}), encoding="utf-8")
    return p


@pytest.fixture()
def annotator() -> AnchorAnnotator:
    return AnchorAnnotator()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_default_config_loads(annotator):
    assert len(annotator._rules) > 0


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AnchorAnnotator(config_path=tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# HEADING block_type — always an anchor
# ---------------------------------------------------------------------------


def test_heading_block_type_detected(annotator):
    block = _tb("Introduction", block_type=BlockType.HEADING)
    result = annotator.annotate_block(block)
    assert result.anchor_type == "heading"
    assert result.anchor_level >= 1


def test_heading_h2_level(annotator):
    block = _tb("## Section Title", block_type=BlockType.HEADING)
    result = annotator.annotate_block(block)
    assert result.anchor_type == "heading"
    assert result.anchor_level == 2


def test_heading_h3_level(annotator):
    block = _tb("### Sub-section", block_type=BlockType.HEADING)
    result = annotator.annotate_block(block)
    assert result.anchor_type == "heading"
    assert result.anchor_level == 3


# ---------------------------------------------------------------------------
# Numbered section detection
# ---------------------------------------------------------------------------


def test_numbered_section_detected(annotator):
    block = _tb("1. Overview of the system")
    result = annotator.annotate_block(block)
    # Could match ordered_list OR numbered_section — both are anchors
    assert result.anchor_type in ("section", "list")
    assert result.anchor_level >= 1


def test_nested_section_number(annotator):
    block = _tb("2.1 Architecture Components")
    result = annotator.annotate_block(block)
    assert result.anchor_type == "section"


# ---------------------------------------------------------------------------
# List item detection
# ---------------------------------------------------------------------------


def test_bullet_list_dash(annotator):
    block = _tb("- First item in the list")
    result = annotator.annotate_block(block)
    assert result.anchor_type == "list"


def test_bullet_list_asterisk(annotator):
    block = _tb("* Another item")
    result = annotator.annotate_block(block)
    assert result.anchor_type == "list"


def test_bullet_list_bullet_char(annotator):
    block = _tb("• Bullet point item")
    result = annotator.annotate_block(block)
    assert result.anchor_type == "list"


# ---------------------------------------------------------------------------
# No anchor — plain paragraph
# ---------------------------------------------------------------------------


def test_plain_paragraph_no_anchor(annotator):
    block = _tb("This is a plain paragraph with no structural markers.")
    result = annotator.annotate_block(block)
    assert result.anchor_type == "none"
    assert result.anchor_level == 0
    assert result.rule_name == ""


# ---------------------------------------------------------------------------
# annotate() — list interface
# ---------------------------------------------------------------------------


def test_annotate_returns_one_per_block(annotator):
    blocks = [_tb(f"Block {i}.") for i in range(5)]
    results = annotator.annotate(blocks)
    assert len(results) == 5


def test_annotate_empty_input(annotator):
    assert annotator.annotate([]) == []


def test_annotate_mixed_blocks(annotator):
    blocks = [
        _tb("Introduction", block_type=BlockType.HEADING),
        _tb("This is a plain paragraph."),
        _tb("- Bullet list item"),
        _tb("2.1 Sub-section title"),
    ]
    results = annotator.annotate(blocks)
    assert results[0].anchor_type == "heading"
    assert results[1].anchor_type == "none"
    assert results[2].anchor_type == "list"
    assert results[3].anchor_type in ("section", "list")


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


def test_custom_rule_matches(tmp_path):
    config = _write_config(tmp_path, [
        {"name": "custom_rule", "type": "section", "pattern": "^CUSTOM:", "level": 1}
    ])
    annotator = AnchorAnnotator(config_path=config)
    block = _tb("CUSTOM: my special section")
    result = annotator.annotate_block(block)
    assert result.anchor_type == "section"
    assert result.rule_name == "custom_rule"


def test_first_matching_rule_wins(tmp_path):
    config = _write_config(tmp_path, [
        {"name": "rule_a", "type": "heading", "pattern": "^##", "level": 2},
        {"name": "rule_b", "type": "section", "pattern": "^##", "level": 1},
    ])
    annotator = AnchorAnnotator(config_path=config)
    block = _tb("## My heading")
    result = annotator.annotate_block(block)
    assert result.rule_name == "rule_a"
