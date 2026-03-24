"""Paragraph block splitter — converts cleaned IRBlocks into hashed TextBlocks.

Oversized blocks (exceeding ``max_tokens``) are split at sentence boundaries
before becoming TextBlocks so that no single chunk ever exceeds the packer's
token budget. This is language-agnostic: it handles both Latin sentence-ending
punctuation (.  !  ?) and CJK full-width equivalents (。 ！ ？).
"""

import hashlib
import logging
import re
import unicodedata

from rag.core.contracts.ir_block import IRBlock
from rag.core.contracts.text_block import TextBlock
from rag.core.interfaces.block_splitter import BaseBlockSplitter

logger = logging.getLogger(__name__)

# Default max tokens per TextBlock before sentence-level splitting kicks in.
# Matches the AnchorAwareChunkPacker default budget so a single block can never
# exceed one chunk on its own.
_DEFAULT_MAX_TOKENS = 512

# Sentence boundary pattern — splits after:
#   . ! ?  (Latin)   followed by whitespace or end-of-string
#   。 ！ ？ (CJK full-width) — these are their own sentence boundary
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+|(?<=[。！？])'
)


def _compute_block_hash(text: str) -> str:
    """Compute a stable SHA-256 hash for a block's canonical text.

    Canonicalisation steps:
    1. Unicode NFC normalisation.
    2. Strip leading/trailing whitespace.
    3. Collapse internal whitespace runs to a single space.

    This ensures the hash is stable across minor formatting differences
    (e.g., double spaces, leading/trailing newlines).

    Args:
        text: Raw block text.

    Returns:
        Hex-encoded SHA-256 digest (64 characters).
    """
    normalised = unicodedata.normalize("NFC", text).strip()
    canonical = " ".join(normalised.split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _count_tokens(text: str) -> int:
    """Estimate token count using tiktoken if available, else char heuristic."""
    try:
        import tiktoken as _tiktoken
        enc = _tiktoken.get_encoding("cl100k_base")
        return max(1, len(enc.encode(text)))
    except Exception:
        return max(1, len(text) // 4)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries.

    Handles both Latin (.!?) and CJK (。！？) sentence endings.
    Empty fragments are discarded.

    Args:
        text: Raw block text.

    Returns:
        List of non-empty sentence strings.
    """
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _sub_split(text: str, max_tokens: int) -> list[str]:
    """Split a text block into sub-blocks each fitting within max_tokens.

    Algorithm:
    1. Split the text into sentences.
    2. Greedily accumulate sentences into a buffer until adding the next
       sentence would exceed max_tokens.
    3. Flush the buffer as a sub-block and start a new one.
    4. If a single sentence exceeds max_tokens on its own, emit it as-is
       rather than truncating (preserving content integrity).

    Args:
        text: Block text that exceeds max_tokens.
        max_tokens: Maximum tokens per sub-block.

    Returns:
        List of sub-block strings, each ideally ≤ max_tokens.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return [text]

    sub_blocks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:
        s_tokens = _count_tokens(sentence)

        if buffer and buffer_tokens + s_tokens > max_tokens:
            # Flush current buffer
            sub_blocks.append(" ".join(buffer))
            buffer = []
            buffer_tokens = 0

        buffer.append(sentence)
        buffer_tokens += s_tokens

    if buffer:
        sub_blocks.append(" ".join(buffer))

    return sub_blocks if sub_blocks else [text]


class ParagraphBlockSplitter(BaseBlockSplitter):
    """Block splitter that maps cleaned IRBlocks to hashed TextBlocks.

    Each IRBlock normally becomes exactly one TextBlock. When a block's token
    count exceeds ``max_tokens``, it is first split at sentence boundaries
    into multiple sub-blocks before being emitted as TextBlocks. This prevents
    oversized chunks regardless of document type (PDF, DOCX, HTML, etc.) or
    language (Latin, CJK, mixed).

    The splitter assigns:
    - a deterministic ``block_hash`` computed from canonicalised text
    - a 0-based ``sequence`` number reflecting position in the document
    - ``doc_id``, ``block_type``, ``page``, and ``section_path`` inherited
      from the source IRBlock

    Blocks with empty text (after stripping) are skipped.

    Usage::

        splitter = ParagraphBlockSplitter(max_tokens=512)
        text_blocks = splitter.split(doc_id="abc123", blocks=ir_blocks)

    Args:
        max_tokens: Maximum token count per output TextBlock. Blocks larger
            than this are sentence-split before emission. Defaults to 512.
    """

    def __init__(self, max_tokens: int = _DEFAULT_MAX_TOKENS) -> None:
        self._max_tokens = max_tokens

    def split(self, doc_id: str, blocks: list[IRBlock]) -> list[TextBlock]:
        """Convert cleaned IRBlocks into sequenced, hashed TextBlocks.

        Blocks exceeding ``max_tokens`` are transparently split at sentence
        boundaries. All resulting sub-blocks inherit the parent block's
        metadata (page, section_path, block_type).

        Args:
            doc_id: Parent document identifier embedded in each TextBlock.
            blocks: Cleaned IRBlocks from the cleaning pipeline.

        Returns:
            Ordered list of TextBlocks. Blocks with empty text are omitted.
            Sequence numbers are contiguous starting from 0.
        """
        result: list[TextBlock] = []
        sequence = 0

        for block in blocks:
            text = block.text.strip()
            if not text:
                continue

            token_count = _count_tokens(text)

            if token_count > self._max_tokens:
                # Sub-split oversized block at sentence boundaries
                sub_blocks = _sub_split(text, self._max_tokens)
                logger.debug(
                    "Block with %d tokens split into %d sub-blocks (max=%d).",
                    token_count, len(sub_blocks), self._max_tokens,
                )
                for sub_text in sub_blocks:
                    if not sub_text:
                        continue
                    result.append(
                        TextBlock(
                            doc_id=doc_id,
                            block_type=block.block_type,
                            text=sub_text,
                            block_hash=_compute_block_hash(sub_text),
                            page=block.page,
                            sequence=sequence,
                            section_path=list(block.section_path),
                        )
                    )
                    sequence += 1
            else:
                result.append(
                    TextBlock(
                        doc_id=doc_id,
                        block_type=block.block_type,
                        text=text,
                        block_hash=_compute_block_hash(text),
                        page=block.page,
                        sequence=sequence,
                        section_path=list(block.section_path),
                    )
                )
                sequence += 1

        return result
