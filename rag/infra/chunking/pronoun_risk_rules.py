"""Pronoun risk detection for chunks — Task 12.1.

Dangling pronouns (it, this, they, their, these, those, he, she, we, that)
reduce the self-containedness of a chunk: a retrieval result containing
"It is important to..." without context leaves the reader unable to resolve
the referent.

``PronounRiskScorer`` assigns a 0.0–1.0 risk score based on the *density*
of opening/dangling pronouns relative to total word count.  Scores are
written into ``chunk.metadata["pronoun_risk"]`` so they can be used by
rerankers, filters, or the evaluation panel.

Score interpretation:
- 0.0 – 0.2  : low risk (chunk is likely self-contained)
- 0.2 – 0.5  : medium risk (some unresolved references possible)
- 0.5 – 1.0  : high risk (chunk heavily relies on surrounding context)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.core.contracts.chunk import Chunk

# ---------------------------------------------------------------------------
# Default pronoun set
# ---------------------------------------------------------------------------

_DEFAULT_PRONOUNS: frozenset[str] = frozenset({
    "it", "its", "itself",
    "this", "these",
    "that", "those",
    "they", "them", "their", "themselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "we", "us", "our", "ours", "ourselves",
})

# Tokeniser: lowercase words only (strips punctuation)
_WORD_RE = re.compile(r"\b[a-z]+\b")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PronounRiskResult:
    """Result of a single pronoun risk scoring run.

    Attributes:
        chunk_id: Identifier of the scored chunk.
        score: Risk score in [0.0, 1.0].  Higher = more dangling pronouns.
        pronoun_count: Number of matched pronoun tokens.
        total_words: Total word count of the chunk text.
        matched_pronouns: List of the actual matched pronoun tokens (for debug).
    """

    chunk_id: str
    score: float
    pronoun_count: int
    total_words: int
    matched_pronouns: list[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        """Human-readable risk level derived from score."""
        if self.score < 0.2:
            return "low"
        if self.score < 0.5:
            return "medium"
        return "high"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class PronounRiskScorer:
    """Score chunks for dangling-pronoun risk.

    Args:
        pronouns: Custom set of pronouns to detect.  Defaults to the
            built-in set of ~20 common English pronouns.
        min_words: Chunks with fewer than this many words receive a score
            of 0.0 (too short to be meaningful).  Defaults to 5.
        annotate: If True, write ``pronoun_risk`` into
            ``chunk.metadata`` on every scored chunk.  Defaults to True.

    Example::

        scorer = PronounRiskScorer()
        results = scorer.score_chunks(chunks)
        high_risk = [r for r in results if r.risk_level == "high"]
    """

    def __init__(
        self,
        pronouns: frozenset[str] | None = None,
        min_words: int = 5,
        annotate: bool = True,
    ) -> None:
        self._pronouns = pronouns if pronouns is not None else _DEFAULT_PRONOUNS
        self._min_words = min_words
        self._annotate = annotate

    def score(self, text: str) -> tuple[float, int, int, list[str]]:
        """Compute the raw pronoun risk score for a text string.

        Args:
            text: Chunk stable_text or any plain string.

        Returns:
            ``(score, pronoun_count, total_words, matched_pronouns)`` tuple.
        """
        words = _WORD_RE.findall(text.lower())
        total = len(words)
        if total < self._min_words:
            return 0.0, 0, total, []

        matched = [w for w in words if w in self._pronouns]
        count = len(matched)
        score = min(1.0, count / total)
        return score, count, total, matched

    def score_chunk(self, chunk: Chunk) -> PronounRiskResult:
        """Score a single chunk and optionally annotate its metadata.

        Args:
            chunk: The chunk to score.

        Returns:
            ``PronounRiskResult`` for this chunk.
        """
        raw_score, count, total, matched = self.score(chunk.stable_text)
        result = PronounRiskResult(
            chunk_id=chunk.chunk_id or chunk.chunk_signature,
            score=raw_score,
            pronoun_count=count,
            total_words=total,
            matched_pronouns=matched,
        )
        if self._annotate:
            # Write into metadata without mutating the original chunk
            chunk.metadata["pronoun_risk"] = round(raw_score, 4)
            chunk.metadata["pronoun_risk_level"] = result.risk_level
        return result

    def score_chunks(self, chunks: list[Chunk]) -> list[PronounRiskResult]:
        """Score a list of chunks.

        Args:
            chunks: Chunks to score.

        Returns:
            One ``PronounRiskResult`` per chunk, in input order.
        """
        return [self.score_chunk(c) for c in chunks]
