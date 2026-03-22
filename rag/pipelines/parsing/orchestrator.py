"""Parser orchestrator — routes artifacts to parsers and handles fallbacks."""

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

from rag.core.contracts.document import Document
from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.core.contracts.parse_report import ParseReport
from rag.core.interfaces.parser import BaseParser
from rag.infra.loading.local_file_loader import RawArtifact
from rag.infra.sniffing.composite_sniffer import SniffResult
from rag.pipelines.parsing.plans import ParsePlan

logger = logging.getLogger(__name__)


def _unsupported_document(artifact: RawArtifact, detected_type: str) -> Document:
    """Produce a Document that signals an unsupported format.

    Args:
        artifact: The raw artifact that could not be parsed.
        detected_type: The detected type string.

    Returns:
        A Document with a single IRBlock containing an error message
        and a ParseReport flagging the unsupported format.
    """
    doc_id = hashlib.sha256(artifact.source_path.encode()).hexdigest()[:16]
    block = IRBlock(
        block_type=BlockType.UNKNOWN,
        text=f"[unsupported_format: {detected_type}]",
    )
    report = ParseReport(
        char_count=0,
        block_count=0,
        non_printable_ratio=0.0,
        repetition_score=0.0,
        parser_used="none",
        fallback_triggered=False,
    )
    return Document(
        doc_id=doc_id,
        source_path=artifact.source_path,
        mime_type=artifact.metadata.get("extension", ""),
        metadata={"unsupported_format": True, "detected_type": detected_type},
        blocks=[block],
        parse_report=report,
    )


class ParserOrchestrator:
    """Routes documents to parsers and manages fallback chains.

    Loads the parser routing table from ``configs/routers/parser_candidates.yaml``
    and uses the registered parser instances to attempt parsing in candidate order.

    Usage::

        registry = {"pymupdf": PyMuPDFParser(), "md_parser": MdParser()}
        orchestrator = ParserOrchestrator(registry)
        plan = orchestrator.route(sniff_result)
        document = orchestrator.parse(artifact, plan)

    Args:
        parser_registry: Dict mapping parser name → BaseParser instance.
        router_config_path: Path to the parser_candidates.yaml config file.
            If None, the default project config is used.
    """

    def __init__(
        self,
        parser_registry: dict[str, BaseParser],
        router_config_path: str | Path | None = None,
    ) -> None:
        self._registry = parser_registry
        self._routes = self._load_routes(router_config_path)

    def _load_routes(self, config_path: str | Path | None) -> dict[str, list[str]]:
        """Load the routing table from YAML.

        Args:
            config_path: Path to parser_candidates.yaml, or None for default.

        Returns:
            Dict mapping detected_type → list of parser names.
        """
        if config_path is None:
            # Walk up from this file to find the project configs/ directory
            candidate = Path(__file__).resolve()
            for parent in candidate.parents:
                yaml_path = parent / "configs" / "routers" / "parser_candidates.yaml"
                if yaml_path.exists():
                    config_path = yaml_path
                    break

        if config_path is None or not Path(config_path).exists():
            raise FileNotFoundError(
                "parser_candidates.yaml not found. Set router_config_path explicitly."
            )

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        routes: dict[str, list[str]] = {}
        for detected_type, route_cfg in data.get("routes", {}).items():
            routes[detected_type] = route_cfg.get("candidates", [])
        return routes

    def route(self, sniff_result: SniffResult) -> ParsePlan:
        """Produce a ParsePlan from a SniffResult.

        Args:
            sniff_result: Output of CompositeSniffer.sniff().

        Returns:
            ParsePlan with the ordered candidate list for this detected type.
        """
        detected_type = sniff_result.detected_type
        candidates = self._routes.get(detected_type, [])
        unsupported = detected_type in ("unsupported", "unknown") or not candidates

        return ParsePlan(
            detected_type=detected_type,
            mime_type=sniff_result.mime_type,
            candidates=candidates,
            unsupported=unsupported,
        )

    def parse(self, artifact: RawArtifact, plan: ParsePlan) -> Document:
        """Parse a RawArtifact using the candidates in the ParsePlan.

        Tries each candidate parser in order. Returns the first successful
        result. If all candidates fail or the plan is unsupported, returns
        an unsupported-format Document.

        Args:
            artifact: RawArtifact from the loader stage.
            plan: ParsePlan produced by ``route()``.

        Returns:
            A Document. If format is unsupported or all parsers fail,
            the Document's metadata will contain ``unsupported_format=True``.
        """
        if plan.unsupported or not plan.candidates:
            logger.warning(
                "Unsupported format '%s' for %s",
                plan.detected_type,
                artifact.source_path,
            )
            return _unsupported_document(artifact, plan.detected_type)

        last_error: Exception | None = None
        fallback_triggered = False

        for i, parser_name in enumerate(plan.candidates):
            parser = self._registry.get(parser_name)
            if parser is None:
                logger.warning("Parser '%s' not registered — skipping.", parser_name)
                continue

            try:
                document = parser.parse(artifact.source_path)
                if i > 0:
                    # A fallback parser was used
                    if document.parse_report:
                        document.parse_report.fallback_triggered = True
                return document
            except Exception as exc:
                logger.warning(
                    "Parser '%s' failed for %s: %s",
                    parser_name,
                    artifact.source_path,
                    exc,
                )
                last_error = exc
                fallback_triggered = True

        logger.error(
            "All parsers failed for %s (last error: %s)",
            artifact.source_path,
            last_error,
        )
        return _unsupported_document(artifact, plan.detected_type)
