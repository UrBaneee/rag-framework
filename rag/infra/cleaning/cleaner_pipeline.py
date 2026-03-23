"""Cleaner pipeline — runs all configured cleaning steps in sequence."""

import logging
from pathlib import Path

import yaml

from rag.core.contracts.ir_block import IRBlock
from rag.core.interfaces.cleaner import BaseCleaner
from rag.infra.cleaning.dedupe_paragraphs import DedupeParagraphs
from rag.infra.cleaning.empty_filter import EmptyBlockFilter
from rag.infra.cleaning.html_nav_footer_remove import HtmlNavFooterRemover
from rag.infra.cleaning.ocr_line_merge import OcrLineMerger
from rag.infra.cleaning.pdf_header_footer_dedupe import PdfHeaderFooterDedupe
from rag.infra.cleaning.unicode_fix import UnicodeFixer

logger = logging.getLogger(__name__)

# Maps step name strings (from YAML) to their factory functions.
# Each factory receives the step config dict and returns a BaseCleaner instance.
_STEP_FACTORIES: dict[str, object] = {}


def _build_steps(step_configs: list[dict]) -> list[BaseCleaner]:
    """Instantiate enabled cleaner steps from config dicts.

    Args:
        step_configs: Ordered list of step config dicts from YAML.

    Returns:
        Ordered list of BaseCleaner instances for enabled steps.

    Raises:
        ValueError: If a step name is not recognised.
    """
    steps: list[BaseCleaner] = []
    for cfg in step_configs:
        name = cfg.get("name", "")
        if not cfg.get("enabled", True):
            logger.debug("Cleaner step '%s' is disabled — skipping.", name)
            continue

        if name == "unicode_fix":
            steps.append(UnicodeFixer())
        elif name == "empty_filter":
            steps.append(EmptyBlockFilter(min_chars=cfg.get("min_chars", 1)))
        elif name == "ocr_line_merge":
            steps.append(OcrLineMerger(short_line_threshold=cfg.get("short_line_threshold", 80)))
        elif name == "html_nav_footer_remove":
            steps.append(HtmlNavFooterRemover())
        elif name == "pdf_header_footer_dedupe":
            steps.append(
                PdfHeaderFooterDedupe(
                    page_fraction_threshold=cfg.get("page_fraction_threshold", 0.5)
                )
            )
        elif name == "dedupe_paragraphs":
            steps.append(DedupeParagraphs())
        else:
            raise ValueError(f"Unknown cleaner step: '{name}'")

    return steps


class CleanerPipeline:
    """Sequential cleaner pipeline that runs all configured steps in order.

    Loads step definitions from ``configs/routers/cleaner_router.yaml``.
    Each enabled step is executed in sequence; the output of one step
    becomes the input of the next.

    Usage::

        pipeline = CleanerPipeline()
        cleaned_blocks = pipeline.run(blocks)

    Args:
        config_path: Path to cleaner_router.yaml. If None, the default
            project config is used (auto-detected by walking up the tree).
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        step_configs = self._load_config(config_path)
        self._steps = _build_steps(step_configs)
        logger.debug(
            "CleanerPipeline initialised with %d step(s): %s",
            len(self._steps),
            [type(s).__name__ for s in self._steps],
        )

    def _load_config(self, config_path: str | Path | None) -> list[dict]:
        """Load the step list from YAML.

        Args:
            config_path: Path to cleaner_router.yaml, or None for default.

        Returns:
            List of step config dicts.

        Raises:
            FileNotFoundError: If the config file cannot be found.
        """
        if config_path is None:
            candidate = Path(__file__).resolve()
            for parent in candidate.parents:
                yaml_path = parent / "configs" / "routers" / "cleaner_router.yaml"
                if yaml_path.exists():
                    config_path = yaml_path
                    break

        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                "cleaner_router.yaml not found. Set config_path explicitly."
            )

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data.get("steps", [])

    @property
    def steps(self) -> list[BaseCleaner]:
        """Return the ordered list of active cleaner steps."""
        return list(self._steps)

    def run(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Run all pipeline steps sequentially on the given blocks.

        Args:
            blocks: Input IRBlocks from the parser stage.

        Returns:
            Cleaned IRBlocks after all steps have been applied.
        """
        current = blocks
        for step in self._steps:
            before = len(current)
            current = step.clean(current)
            after = len(current)
            if before != after:
                logger.debug(
                    "%s: %d → %d blocks", type(step).__name__, before, after
                )
        return current
