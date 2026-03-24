"""End-to-end document ingestion pipeline with optional embedding and indexing."""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml as _yaml
except ImportError:  # pragma: no cover
    _yaml = None  # type: ignore[assignment]

from rag.core.interfaces.doc_store import BaseDocStore
from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.core.interfaces.keyword_index import BaseKeywordIndex
from rag.core.interfaces.trace_store import BaseTraceStore
from rag.core.interfaces.vector_index import BaseVectorIndex
from rag.core.utils.hashing import BlockDiffResult, diff_blocks, fingerprint_bytes, make_doc_id
from rag.infra.cleaning.cleaner_pipeline import CleanerPipeline
from rag.infra.chunking.block_splitter_paragraph import ParagraphBlockSplitter
from rag.infra.chunking.chunk_packer_anchor_aware import AnchorAwareChunkPacker
from rag.infra.loading.local_file_loader import LocalFileLoader
from rag.infra.parsing.html_trafilatura import HtmlTrafilaturaParser
from rag.infra.parsing.md_parser import MdParser
from rag.infra.parsing.pdf_pymupdf import PdfPyMuPDFParser
from rag.infra.sniffing.composite_sniffer import CompositeSniffer
from rag.pipelines.parsing.orchestrator import ParserOrchestrator
from rag.pipelines.parsing.quality_gates import QualityGateChecker

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Summary of a completed ingest run.

    Attributes:
        doc_id: Identifier of the stored document.
        source_path: Absolute path to the ingested file.
        block_count: Number of TextBlocks stored.
        chunk_count: Number of Chunks stored.
        run_id: TraceStore run identifier.
        elapsed_ms: Total wall-clock time in milliseconds.
        skipped: True if the document was already up-to-date and skipped.
        error: Error message if ingestion failed, or None on success.
    """

    doc_id: str
    source_path: str
    block_count: int = 0
    chunk_count: int = 0
    run_id: str = ""
    elapsed_ms: float = 0.0
    embed_tokens: int = 0
    skipped: bool = False
    error: str | None = None
    # Block diff stats — populated when a previous version exists (Task 11.2)
    blocks_added: int = 0
    blocks_removed: int = 0
    blocks_unchanged: int = 0
    diff_available: bool = False


class IngestPipeline:
    """End-to-end document ingestion pipeline.

    Connects all ingestion components in sequence:
    Loader → Sniffer → Router → Parser → Quality Gate →
    Cleaner → Block Splitter → Chunk Packer →
    (optional) Embedder → (optional) BM25 + FAISS indexes →
    DocStore + TraceStore

    When ``embedding_provider`` is supplied, every new chunk is embedded using
    ``stable_text`` and the resulting vectors are stored on the Chunk object.
    When ``vector_index`` or ``keyword_index`` are also provided they are
    updated immediately after embedding. Token usage from the embedding call
    is recorded both in ``IngestResult.embed_tokens`` and in the TraceStore
    run-completion metadata.

    Usage::

        doc_store = SQLiteDocStore("data/rag.db")
        trace_store = SQLiteTraceStore("data/rag.db")
        pipeline = IngestPipeline(doc_store, trace_store)
        result = pipeline.ingest("/path/to/document.pdf")

    Args:
        doc_store: DocStore implementation for persisting documents and chunks.
        trace_store: TraceStore implementation for recording run metadata.
        token_budget: Maximum approximate tokens per chunk. Defaults to 512.
        embedding_provider: Optional provider used to embed chunks.
        vector_index: Optional FAISS (or other) vector index to update.
        keyword_index: Optional BM25 (or other) keyword index to update.
        cleaner_config_path: Path to cleaner_router.yaml. None = auto-detect.
        router_config_path: Path to parser_candidates.yaml. None = auto-detect.
        quality_gate_config_path: Path to quality_gates.yaml. None = auto-detect.
        anchor_config_path: Path to anchors.yaml. None = auto-detect.
    """

    # Default guardrail thresholds (used when no config file is found)
    _DEFAULT_WARN_THRESHOLD: float = 0.5
    _DEFAULT_ERROR_THRESHOLD: float = 0.9

    def __init__(
        self,
        doc_store: BaseDocStore,
        trace_store: BaseTraceStore,
        token_budget: int = 512,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        vector_index: Optional[BaseVectorIndex] = None,
        keyword_index: Optional[BaseKeywordIndex] = None,
        cleaner_config_path: str | Path | None = None,
        router_config_path: str | Path | None = None,
        quality_gate_config_path: str | Path | None = None,
        anchor_config_path: str | Path | None = None,
        resync_config_path: str | Path | None = None,
    ) -> None:
        self._doc_store = doc_store
        self._trace_store = trace_store
        self._embedding_provider = embedding_provider
        self._vector_index = vector_index
        self._keyword_index = keyword_index

        self._loader = LocalFileLoader()
        self._sniffer = CompositeSniffer()

        parser_registry = {
            "pymupdf": PdfPyMuPDFParser(),
            "trafilatura": HtmlTrafilaturaParser(),
            "md_parser": MdParser(),
        }
        self._orchestrator = ParserOrchestrator(
            parser_registry, router_config_path=router_config_path
        )
        self._quality_gate = QualityGateChecker(config_path=quality_gate_config_path)
        self._cleaner = CleanerPipeline(config_path=cleaner_config_path)
        self._splitter = ParagraphBlockSplitter()
        self._packer = AnchorAwareChunkPacker(
            token_budget=token_budget,
            annotator_config_path=anchor_config_path,
        )

        # Load resync guardrail config
        self._warn_threshold, self._error_threshold = self._load_guardrail_config(
            resync_config_path
        )

    def _load_guardrail_config(
        self, config_path: str | Path | None
    ) -> tuple[float, float]:
        """Load warn/error thresholds from resync.yaml, with fallback to defaults.

        Args:
            config_path: Explicit path to resync.yaml, or None for auto-detect.

        Returns:
            ``(warn_threshold, error_threshold)`` tuple.
        """
        if _yaml is None:
            return self._DEFAULT_WARN_THRESHOLD, self._DEFAULT_ERROR_THRESHOLD

        candidates: list[Path] = []
        if config_path is not None:
            candidates.append(Path(config_path))
        else:
            # Auto-detect relative to this file's package root
            pkg_root = Path(__file__).parent.parent
            candidates.append(pkg_root / "configs" / "chunking" / "resync.yaml")
            candidates.append(Path("configs") / "chunking" / "resync.yaml")

        for path in candidates:
            if path.exists():
                try:
                    data = _yaml.safe_load(path.read_text()) or {}
                    guardrails = data.get("guardrails", {})
                    warn = float(guardrails.get("warn_threshold", self._DEFAULT_WARN_THRESHOLD))
                    error = float(guardrails.get("error_threshold", self._DEFAULT_ERROR_THRESHOLD))
                    logger.debug("Loaded resync guardrails from '%s'", path)
                    return warn, error
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to load resync config '%s': %s", path, exc)

        return self._DEFAULT_WARN_THRESHOLD, self._DEFAULT_ERROR_THRESHOLD

    def _check_resync_guardrails(
        self,
        block_diff: "BlockDiffResult",
        source_path: str,
        run_id: str,
    ) -> None:
        """Emit log warnings and trace events when change ratio exceeds thresholds.

        Args:
            block_diff: Populated BlockDiffResult from the current ingest run.
            source_path: Absolute path to the source file (for log messages).
            run_id: Current TraceStore run identifier.
        """
        total = block_diff.unchanged_count + block_diff.added_count + block_diff.removed_count
        if total == 0:
            return

        changed = block_diff.added_count + block_diff.removed_count
        ratio = changed / total

        if ratio >= self._error_threshold:
            logger.error(
                "Resync guardrail ERROR: changed ratio %.2f >= %.2f for '%s' "
                "(+%d added, -%d removed, %d unchanged)",
                ratio, self._error_threshold, source_path,
                block_diff.added_count, block_diff.removed_count, block_diff.unchanged_count,
            )
            self._trace_store.save_run(
                run_type="resync_threshold_exceeded",
                metadata={
                    "parent_run_id": run_id,
                    "level": "error",
                    "changed_ratio": ratio,
                    "threshold": self._error_threshold,
                    "blocks_added": block_diff.added_count,
                    "blocks_removed": block_diff.removed_count,
                    "blocks_unchanged": block_diff.unchanged_count,
                    "source_path": source_path,
                },
            )
        elif ratio >= self._warn_threshold:
            logger.warning(
                "Resync guardrail WARNING: changed ratio %.2f >= %.2f for '%s' "
                "(+%d added, -%d removed, %d unchanged)",
                ratio, self._warn_threshold, source_path,
                block_diff.added_count, block_diff.removed_count, block_diff.unchanged_count,
            )
            self._trace_store.save_run(
                run_type="resync_threshold_exceeded",
                metadata={
                    "parent_run_id": run_id,
                    "level": "warning",
                    "changed_ratio": ratio,
                    "threshold": self._warn_threshold,
                    "blocks_added": block_diff.added_count,
                    "blocks_removed": block_diff.removed_count,
                    "blocks_unchanged": block_diff.unchanged_count,
                    "source_path": source_path,
                },
            )

    def ingest(self, source_path: str | Path) -> IngestResult:
        """Ingest a single document file end-to-end.

        Args:
            source_path: Path to the file to ingest.

        Returns:
            IngestResult summarising the outcome.
        """
        source_path = str(Path(source_path).resolve())
        start = time.monotonic()

        run_id = self._trace_store.save_run(
            run_type="ingest",
            metadata={"source_path": source_path},
        )

        try:
            result = self._run(source_path, run_id)
        except Exception as exc:
            logger.exception("Ingest failed for '%s': %s", source_path, exc)
            result = IngestResult(
                doc_id="",
                source_path=source_path,
                run_id=run_id,
                error=str(exc),
            )

        result.elapsed_ms = (time.monotonic() - start) * 1000
        self._trace_store.save_run(
            run_type="ingest_complete",
            metadata={
                "source_path": source_path,
                "run_id": run_id,
                "doc_id": result.doc_id,
                "block_count": result.block_count,
                "chunk_count": result.chunk_count,
                "embed_tokens": result.embed_tokens,
                "elapsed_ms": result.elapsed_ms,
                "error": result.error,
            },
        )
        return result

    def _run(self, source_path: str, run_id: str) -> IngestResult:
        """Internal pipeline execution — all stages.

        Args:
            source_path: Resolved absolute path to the source file.
            run_id: TraceStore run identifier.

        Returns:
            Populated IngestResult on success.

        Raises:
            Exception: On any unrecoverable pipeline error.
        """
        # 1. Load
        artifact = self._loader.load(source_path)

        # 1b. Fingerprint — detect unchanged documents before deep processing
        content_hash = fingerprint_bytes(artifact.raw_bytes)
        doc_id = make_doc_id(source_path, content_hash)

        if self._doc_store.document_exists(doc_id):
            logger.info(
                "Skipping unchanged document '%s' (doc_id=%s)", source_path, doc_id
            )
            return IngestResult(
                doc_id=doc_id,
                source_path=source_path,
                run_id=run_id,
                skipped=True,
            )

        # 2. Sniff
        sniff_result = self._sniffer.sniff(artifact)
        logger.debug("Detected type: %s", sniff_result.detected_type)

        # 3. Route + Parse
        plan = self._orchestrator.route(sniff_result)
        document = self._orchestrator.parse(artifact, plan)

        # 4. Quality gate
        if document.parse_report:
            gate_result = self._quality_gate.check(document.parse_report)
            if not gate_result.passed:
                logger.warning(
                    "Quality gate failed for '%s': %s",
                    source_path,
                    "; ".join(gate_result.reasons),
                )

        # Override parser's doc_id with fingerprint-based id so the DocStore
        # lookup in step 1b is consistent across runs.
        document = document.model_copy(update={
            "doc_id": doc_id,
            "metadata": {**document.metadata, "content_hash": content_hash},
        })

        # 5. Store document
        self._doc_store.save_document(document)

        # 6. Clean
        cleaned_blocks = self._cleaner.run(document.blocks)

        # 7. Split into TextBlocks
        text_blocks = self._splitter.split(doc_id, cleaned_blocks)

        # 7b. Block diff — compare new blocks against the previous version
        block_diff: BlockDiffResult | None = None
        if hasattr(self._doc_store, "get_prev_blocks_for_source"):
            prev_blocks = self._doc_store.get_prev_blocks_for_source(source_path)  # type: ignore[attr-defined]
            if prev_blocks:
                old_hashes = [b.block_hash for b in prev_blocks]
                new_hashes = [b.block_hash for b in text_blocks]
                block_diff = diff_blocks(old_hashes, new_hashes)
                logger.debug(
                    "Block diff for '%s': +%d added, -%d removed, %d unchanged",
                    source_path,
                    block_diff.added_count,
                    block_diff.removed_count,
                    block_diff.unchanged_count,
                )
                # 7c. Guardrail check — warn/error if change ratio exceeds thresholds
                self._check_resync_guardrails(block_diff, source_path, run_id)

        self._doc_store.save_text_blocks(text_blocks)

        # 8. Pack into Chunks
        chunks = self._packer.pack(text_blocks)

        # Assign stable chunk_ids from chunk_signature so embedding and
        # indexing work before (and consistently with) DocStore persistence.
        chunks = [
            c.model_copy(update={"chunk_id": c.chunk_id or c.chunk_signature})
            for c in chunks
        ]

        # 9. Embed + index (optional)
        embed_tokens = 0
        if self._embedding_provider is not None and chunks:
            texts = [c.stable_text for c in chunks]
            embed_result = self._embedding_provider.embed(texts)
            embed_tokens = getattr(embed_result, "prompt_tokens", 0)

            # Vectors come back as a plain list when embed() is used directly;
            # use embed_with_usage() if the provider supports it for token counts.
            if hasattr(self._embedding_provider, "embed_with_usage"):
                full_result = self._embedding_provider.embed_with_usage(texts)
                vectors = full_result.vectors
                embed_tokens = full_result.prompt_tokens
            else:
                vectors = embed_result if isinstance(embed_result, list) else embed_result.vectors

            # Attach vectors to chunks (create updated copies via model_copy)
            chunks = [
                c.model_copy(update={"embedding": vec})
                for c, vec in zip(chunks, vectors)
            ]
            logger.debug(
                "Embedded %d chunks using %d tokens.", len(chunks), embed_tokens
            )

            if self._keyword_index is not None:
                self._keyword_index.add(chunks)

            if self._vector_index is not None:
                self._vector_index.add(chunks)

        self._doc_store.save_chunks(chunks)

        logger.info(
            "Ingested '%s': %d blocks, %d chunks, %d embed tokens (run=%s)",
            source_path,
            len(text_blocks),
            len(chunks),
            embed_tokens,
            run_id,
        )

        return IngestResult(
            doc_id=doc_id,
            source_path=source_path,
            block_count=len(text_blocks),
            chunk_count=len(chunks),
            run_id=run_id,
            embed_tokens=embed_tokens,
            skipped=False,
            blocks_added=block_diff.added_count if block_diff else 0,
            blocks_removed=block_diff.removed_count if block_diff else 0,
            blocks_unchanged=block_diff.unchanged_count if block_diff else 0,
            diff_available=block_diff is not None,
        )
