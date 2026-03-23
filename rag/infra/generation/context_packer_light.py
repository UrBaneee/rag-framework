"""Light context packer — selects and deduplicates chunks for LLM context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from rag.core.contracts.candidate import Candidate
from rag.core.contracts.citation import Citation

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 3
_DEFAULT_TOKEN_BUDGET = 2048


@dataclass
class PackedContext:
    """Result of context packing: selected candidates and their citation map.

    Attributes:
        candidates: Ordered list of selected, deduplicated candidates.
        citations: List of Citation objects (1-based ref_number) in the
            same order as ``candidates``.
        context_text: Pre-formatted context block ready for prompt injection.
            Each entry is formatted as ``[N] <display_text>``.
        total_tokens: Estimated token count of the packed context.
        truncated: True when the token budget was reached before all
            top-k candidates were included.
    """

    candidates: list[Candidate] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    context_text: str = ""
    total_tokens: int = 0
    truncated: bool = False


class LightContextPacker:
    """Selects up to ``top_k`` unique chunks within a token budget.

    Processing steps:
    1. Iterate ranked candidates in order (highest final_score first).
    2. Skip any candidate whose ``stable_text`` (stripped) has already
       been seen — this catches exact duplicates introduced by hybrid
       retrieval returning the same chunk via both BM25 and vector paths.
    3. Stop when ``top_k`` unique candidates have been collected OR the
       cumulative token count would exceed ``token_budget``.
    4. Assign 1-based ``ref_number`` values and build Citation objects.
    5. Assemble ``context_text`` as a numbered list of ``display_text`` values.

    Token estimation uses ``Chunk.token_count`` if available, otherwise
    falls back to ``len(display_text) // 4`` (approx 1 token = 4 chars).

    Args:
        top_k: Maximum number of unique chunks to include. Defaults to 3.
        token_budget: Maximum total tokens across all packed chunks.
            Defaults to 2048.
    """

    def __init__(
        self,
        top_k: int = _DEFAULT_TOP_K,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if token_budget < 1:
            raise ValueError(f"token_budget must be >= 1, got {token_budget}")
        self._top_k = top_k
        self._token_budget = token_budget

    # ── Public API ────────────────────────────────────────────────────────────

    def pack(self, candidates: list[Candidate]) -> PackedContext:
        """Pack ranked candidates into a context window.

        Args:
            candidates: Ranked list of Candidate objects (best first).

        Returns:
            PackedContext with deduplicated candidates, citations, and
            formatted context text.
        """
        selected: list[Candidate] = []
        seen_texts: set[str] = set()
        total_tokens = 0
        truncated = False

        for candidate in candidates:
            if len(selected) >= self._top_k:
                break

            # Deduplicate on stripped stable_text
            dedup_key = candidate.stable_text.strip()
            if dedup_key in seen_texts:
                logger.debug(
                    "context_packer: skipping duplicate chunk_id=%s", candidate.chunk_id
                )
                continue
            seen_texts.add(dedup_key)

            # Check token budget
            chunk_tokens = self._estimate_tokens(candidate)
            if total_tokens + chunk_tokens > self._token_budget:
                truncated = True
                logger.debug(
                    "context_packer: token budget %d exhausted at chunk_id=%s "
                    "(would add %d tokens, current=%d)",
                    self._token_budget,
                    candidate.chunk_id,
                    chunk_tokens,
                    total_tokens,
                )
                break

            selected.append(candidate)
            total_tokens += chunk_tokens

        citations = self._build_citations(selected)
        context_text = self._format_context(selected)

        logger.debug(
            "context_packer: packed %d/%d candidates, tokens=%d, truncated=%s",
            len(selected),
            len(candidates),
            total_tokens,
            truncated,
        )

        return PackedContext(
            candidates=selected,
            citations=citations,
            context_text=context_text,
            total_tokens=total_tokens,
            truncated=truncated,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _estimate_tokens(candidate: Candidate) -> int:
        """Estimate token count for a candidate.

        Prefers metadata ``token_count`` key; falls back to display_text heuristic.
        """
        token_count = candidate.metadata.get("token_count", 0)
        if isinstance(token_count, int) and token_count > 0:
            return token_count
        # Heuristic: ~1 token per 4 characters
        return max(1, len(candidate.display_text) // 4)

    @staticmethod
    def _build_citations(candidates: list[Candidate]) -> list[Citation]:
        """Build 1-based Citation objects for selected candidates."""
        citations: list[Citation] = []
        for ref_number, candidate in enumerate(candidates, start=1):
            source_label = candidate.metadata.get("source_label", "")
            if not source_label:
                # Build a reasonable label from doc_id and page
                page = candidate.metadata.get("start_page") or candidate.metadata.get("page")
                source_label = candidate.doc_id
                if page is not None:
                    source_label = f"{candidate.doc_id} — page {page}"

            display_excerpt = candidate.display_text[:120].rstrip()
            if len(candidate.display_text) > 120:
                display_excerpt += "…"

            citations.append(
                Citation(
                    ref_number=ref_number,
                    chunk_id=candidate.chunk_id,
                    doc_id=candidate.doc_id,
                    source_label=source_label,
                    page=candidate.metadata.get("start_page") or candidate.metadata.get("page"),
                    display_text=display_excerpt,
                )
            )
        return citations

    @staticmethod
    def _format_context(candidates: list[Candidate]) -> str:
        """Format candidates as a numbered context block.

        Each entry: ``[N] <display_text>`` separated by blank lines.
        """
        if not candidates:
            return ""
        parts: list[str] = []
        for ref_number, candidate in enumerate(candidates, start=1):
            parts.append(f"[{ref_number}] {candidate.display_text.strip()}")
        return "\n\n".join(parts)
