"""Anchor-aware chunk packer — packs TextBlocks into Chunks respecting token budgets."""

import hashlib
import logging
from pathlib import Path
from typing import Any

from rag.core.contracts.chunk import Chunk
from rag.core.contracts.text_block import TextBlock
from rag.core.interfaces.chunk_packer import BaseChunkPacker
from rag.infra.chunking.anchor_annotator_rules import AnchorAnnotation, AnchorAnnotator

logger = logging.getLogger(__name__)

# Default token budget per chunk
_DEFAULT_TOKEN_BUDGET = 512

# Tiktoken encoder — used for accurate token counting across all scripts
# (English, Chinese, Japanese, etc.). Falls back to char-based estimate
# if tiktoken is unavailable.
try:
    import tiktoken as _tiktoken

    _ENCODER = _tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_AVAILABLE = True
except Exception:  # pragma: no cover
    _ENCODER = None
    _TIKTOKEN_AVAILABLE = False

# Fallback: chars per token for English-only heuristic
_CHARS_PER_TOKEN = 4


def _approx_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base) or fall back to char heuristic.

    tiktoken gives accurate counts for all scripts including CJK where each
    character is typically 1 token, vs ~0.25 tokens in the naïve 4-chars
    heuristic. Without it a 1000-char Chinese document is underestimated by
    4× and never gets split into multiple chunks.

    Args:
        text: Input text.

    Returns:
        Token count (exact if tiktoken available, approximate otherwise).
    """
    if _TIKTOKEN_AVAILABLE and _ENCODER is not None:
        return max(1, len(_ENCODER.encode(text)))
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _compute_chunk_signature(block_hashes: list[str]) -> str:
    """Compute SHA-256 over the ordered block hashes.

    Args:
        block_hashes: Ordered list of block_hash strings.

    Returns:
        64-character hex digest.
    """
    combined = "|".join(block_hashes)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _build_chunk(doc_id: str, blocks: list[TextBlock], annotations: list[AnchorAnnotation]) -> Chunk:
    """Assemble a Chunk from a group of consecutive TextBlocks.

    Args:
        doc_id: Parent document ID.
        blocks: Consecutive TextBlocks forming this chunk.
        annotations: Anchor annotations parallel to blocks.

    Returns:
        Assembled Chunk with stable_text, display_text, chunk_signature.
    """
    parts: list[str] = []
    display_parts: list[str] = []

    for block, ann in zip(blocks, annotations):
        text = block.text.strip()
        if ann.anchor_type != "none":
            # Prefix display text with anchor context for UI readability
            display_parts.append(f"[{ann.anchor_type.upper()}] {text}")
        else:
            display_parts.append(text)
        parts.append(text)

    stable_text = "\n\n".join(parts)
    display_text = "\n\n".join(display_parts)
    block_hashes = [b.block_hash for b in blocks]
    chunk_signature = _compute_chunk_signature(block_hashes)
    token_count = _approx_tokens(stable_text)

    # Collect metadata from the first block
    metadata: dict[str, Any] = {}
    first = blocks[0]
    if first.page is not None:
        metadata["start_page"] = first.page
    last = blocks[-1]
    if last.page is not None:
        metadata["end_page"] = last.page
    if first.section_path:
        metadata["section_path"] = first.section_path
    metadata["loc_span"] = [first.sequence, last.sequence]

    return Chunk(
        doc_id=doc_id,
        stable_text=stable_text,
        display_text=display_text,
        chunk_signature=chunk_signature,
        block_hashes=block_hashes,
        token_count=token_count,
        metadata=metadata,
    )


class AnchorAwareChunkPacker(BaseChunkPacker):
    """Chunk packer that respects anchor boundaries and token budgets.

    Packing strategy:
    1. Annotate all blocks with anchor information.
    2. Iterate blocks in sequence. Start a new chunk when:
       - The current block is an anchor (heading/section) AND the buffer
         is non-empty (anchor = natural break point).
       - Adding the current block would exceed the token budget.
    3. Build each chunk from the accumulated buffer.

    This produces chunks that start at natural section boundaries and
    stay within the token budget.

    Usage::

        packer = AnchorAwareChunkPacker(token_budget=512)
        chunks = packer.pack(text_blocks)

    Args:
        token_budget: Maximum approximate token count per chunk.
            Defaults to 512.
        annotator_config_path: Path to anchors.yaml. If None, auto-detected.
    """

    def __init__(
        self,
        token_budget: int = _DEFAULT_TOKEN_BUDGET,
        annotator_config_path: str | Path | None = None,
    ) -> None:
        self._token_budget = token_budget
        self._annotator = AnchorAnnotator(config_path=annotator_config_path)

    def pack(self, blocks: list[TextBlock]) -> list[Chunk]:
        """Pack TextBlocks into anchor-aware Chunks.

        Args:
            blocks: Ordered TextBlocks from the block splitter.
                Must all share the same doc_id.

        Returns:
            List of Chunks. Empty input → empty output.
        """
        if not blocks:
            return []

        doc_id = blocks[0].doc_id
        annotations = self._annotator.annotate(blocks)

        chunks: list[Chunk] = []
        buffer_blocks: list[TextBlock] = []
        buffer_anns: list[AnchorAnnotation] = []
        buffer_tokens = 0

        for block, ann in zip(blocks, annotations):
            block_tokens = _approx_tokens(block.text)
            is_anchor = ann.anchor_type != "none"

            # Flush buffer at anchor boundaries or when budget would overflow
            if buffer_blocks and (
                (is_anchor) or (buffer_tokens + block_tokens > self._token_budget)
            ):
                chunks.append(_build_chunk(doc_id, buffer_blocks, buffer_anns))
                buffer_blocks = []
                buffer_anns = []
                buffer_tokens = 0

            buffer_blocks.append(block)
            buffer_anns.append(ann)
            buffer_tokens += block_tokens

        # Flush remaining buffer
        if buffer_blocks:
            chunks.append(_build_chunk(doc_id, buffer_blocks, buffer_anns))

        return chunks
