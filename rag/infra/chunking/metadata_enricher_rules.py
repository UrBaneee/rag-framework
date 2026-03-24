"""Rules-based metadata enrichment for chunks — Task 12.2.

Adds ``title``, ``summary``, and ``tags`` to chunk metadata using
deterministic heuristic rules — no LLM required.  This provides a fast,
zero-cost baseline that can be overridden by the LLM enricher (Task 12.3).

Rules applied
-------------
title
    1. First heading line starting with ``#`` (Markdown) in the chunk text.
    2. First non-empty sentence (up to 80 characters).
    3. First 60 characters of the stable_text as a last resort.

summary
    First two sentences of the chunk's stable_text, truncated to
    ``max_summary_chars`` (default 256).

tags
    - ``has_code``: chunk contains a fenced code block or inline backticks.
    - ``has_table``: chunk contains a Markdown table (``|`` rows).
    - ``has_list``: chunk contains bullet/numbered list markers.
    - ``high_pronoun_risk``: ``pronoun_risk`` metadata >= 0.5 (set by
      PronounRiskScorer if it ran first).
    - Section path tokens from ``chunk.metadata.get("section_path", [])``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.core.contracts.chunk import Chunk

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
_CODE_FENCE_RE = re.compile(r"```|`[^`]+`")
_TABLE_ROW_RE = re.compile(r"^\|.+\|", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^(\s*[-*+]|\s*\d+\.)\s+", re.MULTILINE)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EnrichmentResult:
    """Result of a rules-based enrichment pass on a single chunk.

    Attributes:
        chunk_id: Identifier of the enriched chunk.
        title: Derived title string.
        summary: Short summary string.
        tags: List of tag strings.
        fields_set: Names of metadata fields that were written.
    """

    chunk_id: str
    title: str
    summary: str
    tags: list[str] = field(default_factory=list)
    fields_set: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Enricher
# ---------------------------------------------------------------------------


class RulesMetadataEnricher:
    """Enrich chunk metadata with title, summary, and tags using heuristic rules.

    Args:
        max_summary_chars: Maximum length of the generated summary.
            Defaults to 256.
        overwrite: If False (default), skip fields already present in
            ``chunk.metadata`` so existing values are preserved.
        annotate: If True (default), write results directly into
            ``chunk.metadata``.

    Example::

        enricher = RulesMetadataEnricher()
        results = enricher.enrich_chunks(chunks)
        for r in results:
            print(r.chunk_id, r.title, r.tags)
    """

    def __init__(
        self,
        max_summary_chars: int = 256,
        overwrite: bool = False,
        annotate: bool = True,
    ) -> None:
        self._max_summary_chars = max_summary_chars
        self._overwrite = overwrite
        self._annotate = annotate

    # ── Internal helpers ────────────────────────────────────────────────────

    def _extract_title(self, text: str) -> str:
        """Derive a title from chunk text."""
        # 1. Markdown heading
        match = _HEADING_RE.search(text)
        if match:
            return match.group(1).strip()[:120]
        # 2. First non-empty sentence ≤ 80 chars
        sentences = _SENTENCE_END_RE.split(text.strip())
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) <= 80:
                return sent
        # 3. First 60 chars
        return text.strip()[:60].replace("\n", " ")

    def _extract_summary(self, text: str) -> str:
        """Extract a short summary from the first two sentences."""
        sentences = _SENTENCE_END_RE.split(text.strip())
        summary = " ".join(s.strip() for s in sentences[:2] if s.strip())
        return summary[: self._max_summary_chars]

    def _extract_tags(self, chunk: Chunk) -> list[str]:
        """Derive structural tags from the chunk text and metadata."""
        text = chunk.stable_text
        tags: list[str] = []

        if _CODE_FENCE_RE.search(text):
            tags.append("has_code")
        if _TABLE_ROW_RE.search(text):
            tags.append("has_table")
        if _LIST_ITEM_RE.search(text):
            tags.append("has_list")

        pronoun_risk = chunk.metadata.get("pronoun_risk", 0.0)
        if isinstance(pronoun_risk, (int, float)) and pronoun_risk >= 0.5:
            tags.append("high_pronoun_risk")

        section_path = chunk.metadata.get("section_path", [])
        if isinstance(section_path, list):
            for token in section_path:
                if isinstance(token, str) and token.strip():
                    tags.append(f"section:{token.strip()}")

        return tags

    # ── Public API ──────────────────────────────────────────────────────────

    def enrich(self, chunk: Chunk) -> EnrichmentResult:
        """Enrich a single chunk and optionally write results to its metadata.

        Args:
            chunk: The chunk to enrich.

        Returns:
            ``EnrichmentResult`` with derived title, summary, and tags.
        """
        text = chunk.stable_text
        fields_set: list[str] = []

        title = self._extract_title(text)
        summary = self._extract_summary(text)
        tags = self._extract_tags(chunk)

        if self._annotate:
            if self._overwrite or "title" not in chunk.metadata:
                chunk.metadata["title"] = title
                fields_set.append("title")
            if self._overwrite or "summary" not in chunk.metadata:
                chunk.metadata["summary"] = summary
                fields_set.append("summary")
            if self._overwrite or "tags" not in chunk.metadata:
                chunk.metadata["tags"] = tags
                fields_set.append("tags")

        return EnrichmentResult(
            chunk_id=chunk.chunk_id or chunk.chunk_signature,
            title=title,
            summary=summary,
            tags=tags,
            fields_set=fields_set,
        )

    def enrich_chunks(self, chunks: list[Chunk]) -> list[EnrichmentResult]:
        """Enrich a list of chunks.

        Args:
            chunks: Chunks to enrich.

        Returns:
            One ``EnrichmentResult`` per chunk, in input order.
        """
        return [self.enrich(c) for c in chunks]
