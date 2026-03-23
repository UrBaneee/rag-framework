"""Grounded prompt builder — assembles the LLM prompt for evidence-based QA."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag.core.contracts.citation import Citation
from rag.infra.generation.context_packer_light import PackedContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System instructions (static)
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS = """\
You are a precise question-answering assistant. Answer the user's question
using ONLY the numbered context passages provided below. Follow these rules:

1. GROUNDED ANSWERS ONLY — every factual claim must be supported by a context
   passage. Add an inline citation marker [N] immediately after each claim,
   where N is the passage number.

2. MULTIPLE SOURCES — if several passages support a claim, cite all of them,
   e.g. [1][3].

3. INSUFFICIENT EVIDENCE — if the provided context does not contain enough
   information to answer the question, respond with exactly:
   "I don't have enough information in the provided context to answer this question."
   Do not guess, infer beyond the evidence, or use outside knowledge.

4. CONCISE AND DIRECT — answer in plain prose. Do not repeat the question.
   Do not add preamble like "Based on the context…".
"""

_CITATION_KEY_HEADER = "=== Source Index ===\n"
_CONTEXT_HEADER = "=== Context Passages ===\n"
_QUESTION_HEADER = "=== Question ===\n"


@dataclass
class BuiltPrompt:
    """A fully assembled prompt ready for LLM submission.

    Attributes:
        system: System-level instructions (sent as system role if supported).
        user: User-turn content combining context, citation map, and question.
        full_text: Concatenation of system + user for single-turn models.
    """

    system: str
    user: str

    @property
    def full_text(self) -> str:
        """Combined system + user text for single-turn (completion) models."""
        return f"{self.system}\n\n{self.user}"


class GroundedPromptBuilder:
    """Builds a grounded QA prompt from a query and packed context.

    The prompt structure is:

    .. code-block:: text

        [system instructions]

        === Source Index ===
        [1] doc.pdf — page 3
        [2] report.pdf — page 11

        === Context Passages ===
        [1] <display_text>

        [2] <display_text>

        === Question ===
        <query>

    The system instructions emphasise:
    - Answer only from the numbered passages
    - Cite each claim with inline [N] markers
    - Abstain with a fixed phrase when evidence is insufficient

    Args:
        system_instructions: Override the default system instructions block.
            Useful for domain-specific grounding rules.
    """

    def __init__(self, system_instructions: str | None = None) -> None:
        self._system = system_instructions or _SYSTEM_INSTRUCTIONS.strip()

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, query: str, packed: PackedContext) -> BuiltPrompt:
        """Assemble the LLM prompt.

        Args:
            query: The user's natural-language question.
            packed: PackedContext produced by LightContextPacker.

        Returns:
            BuiltPrompt with ``system`` and ``user`` fields populated.
        """
        user = self._build_user_turn(query, packed.citations, packed.context_text)
        logger.debug(
            "prompt_builder: query_len=%d context_len=%d citations=%d",
            len(query),
            len(packed.context_text),
            len(packed.citations),
        )
        return BuiltPrompt(system=self._system, user=user)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_user_turn(
        query: str,
        citations: list[Citation],
        context_text: str,
    ) -> str:
        """Construct the user-turn string."""
        parts: list[str] = []

        # Citation key / source index
        if citations:
            lines = [_CITATION_KEY_HEADER]
            for cit in citations:
                lines.append(f"[{cit.ref_number}] {cit.source_label}")
            parts.append("\n".join(lines))

        # Context passages
        if context_text:
            parts.append(f"{_CONTEXT_HEADER}{context_text}")
        else:
            parts.append(
                f"{_CONTEXT_HEADER}(No context passages available — "
                "please respond with the abstain message.)"
            )

        # Question
        parts.append(f"{_QUESTION_HEADER}{query.strip()}")

        return "\n\n".join(parts)
