"""Tests for the CleanerPipeline — wires all 6 cleaner steps in sequence."""

from pathlib import Path

import pytest
import yaml

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.infra.cleaning.cleaner_pipeline import CleanerPipeline


def _block(text: str, page: int | None = None) -> IRBlock:
    return IRBlock(block_type=BlockType.PARAGRAPH, text=text, page=page)


def _write_config(tmp_path: Path, steps: list[dict]) -> Path:
    p = tmp_path / "cleaner_router.yaml"
    p.write_text(yaml.dump({"steps": steps}), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_default_config_loads():
    pipeline = CleanerPipeline()
    assert len(pipeline.steps) == 6  # all steps enabled by default


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CleanerPipeline(config_path=tmp_path / "nonexistent.yaml")


def test_unknown_step_raises(tmp_path):
    config = _write_config(tmp_path, [{"name": "does_not_exist", "enabled": True}])
    with pytest.raises(ValueError, match="Unknown cleaner step"):
        CleanerPipeline(config_path=config)


def test_disabled_step_skipped(tmp_path):
    config = _write_config(tmp_path, [
        {"name": "unicode_fix", "enabled": False},
        {"name": "empty_filter", "enabled": True},
    ])
    pipeline = CleanerPipeline(config_path=config)
    assert len(pipeline.steps) == 1


# ---------------------------------------------------------------------------
# run() — individual step effects
# ---------------------------------------------------------------------------


def test_run_unicode_fix_applied(tmp_path):
    config = _write_config(tmp_path, [{"name": "unicode_fix", "enabled": True}])
    pipeline = CleanerPipeline(config_path=config)
    blocks = [_block("\u201cHello\u201d")]
    result = pipeline.run(blocks)
    assert result[0].text == '"Hello"'


def test_run_empty_filter_applied(tmp_path):
    config = _write_config(tmp_path, [{"name": "empty_filter", "enabled": True, "min_chars": 1}])
    pipeline = CleanerPipeline(config_path=config)
    blocks = [_block("Real content."), _block(""), _block("   ")]
    result = pipeline.run(blocks)
    assert len(result) == 1


def test_run_dedupe_applied(tmp_path):
    config = _write_config(tmp_path, [{"name": "dedupe_paragraphs", "enabled": True}])
    pipeline = CleanerPipeline(config_path=config)
    blocks = [_block("Same text."), _block("Same text.")]
    result = pipeline.run(blocks)
    assert len(result) == 1


def test_run_pdf_header_footer_applied(tmp_path):
    config = _write_config(tmp_path, [
        {"name": "pdf_header_footer_dedupe", "enabled": True, "page_fraction_threshold": 0.5}
    ])
    pipeline = CleanerPipeline(config_path=config)
    blocks = [
        _block("Page Header", page=1),
        _block("Content A", page=1),
        _block("Page Header", page=2),
        _block("Content B", page=2),
        _block("Page Header", page=3),
        _block("Content C", page=3),
    ]
    result = pipeline.run(blocks)
    texts = [b.text for b in result]
    assert "Page Header" not in texts
    assert "Content A" in texts


# ---------------------------------------------------------------------------
# run() — full pipeline on a small document
# ---------------------------------------------------------------------------


def test_run_full_pipeline_cleans_document():
    """Run a messy document through the complete default pipeline."""
    pipeline = CleanerPipeline()

    blocks = [
        # Unicode issues — page 1
        _block("\u201cIntroduction\u201d to the topic.", page=1),
        # Empty block — page 1
        _block("", page=1),
        # Nav boilerplate — page 1
        _block("Home | About | Contact | Login", page=1),
        # Repeating header (all 3 pages → 100% → removed)
        _block("Running Title", page=1),
        _block("Unique content on page one here.", page=1),
        _block("Running Title", page=2),
        _block("Unique content on page two here.", page=2),
        _block("Running Title", page=3),
        _block("Unique content on page three here.", page=3),
        # Exact duplicate of an earlier block on the same page
        _block("Unique content on page three here.", page=3),
    ]

    result = pipeline.run(blocks)
    texts = [b.text for b in result]

    # Unicode fixed
    assert '"Introduction" to the topic.' in texts
    # Empty removed
    assert "" not in texts
    # Nav removed
    assert "Home | About | Contact | Login" not in texts
    # Running header removed
    assert "Running Title" not in texts
    # Unique content kept
    assert "Unique content on page one here." in texts
    assert "Unique content on page two here." in texts
    # Deduplicated (appears once despite two blocks)
    assert texts.count("Unique content on page three here.") == 1


def test_run_empty_input():
    pipeline = CleanerPipeline()
    assert pipeline.run([]) == []


def test_run_preserves_clean_blocks():
    pipeline = CleanerPipeline()
    # Use None page so pdf_header_footer_dedupe is a no-op
    blocks = [
        _block("Clean paragraph one."),
        _block("Clean paragraph two."),
    ]
    result = pipeline.run(blocks)
    assert len(result) == 2
