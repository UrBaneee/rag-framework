"""OCR line-merge cleaner — reconstructs paragraphs from OCR-fragmented lines."""

from rag.core.contracts.ir_block import IRBlock
from rag.core.interfaces.cleaner import BaseCleaner

# A line is considered a continuation of the previous paragraph (not a new
# one) if it is shorter than this character count. Typical OCR fragments a
# paragraph into many short lines; full sentences are longer.
_DEFAULT_SHORT_LINE_THRESHOLD = 80

# Sentence-ending punctuation. A line ending with these characters is treated
# as a complete sentence and is not merged with the next line.
_SENTENCE_ENDINGS = frozenset(".!?\"'")


def _ends_sentence(line: str) -> bool:
    """Return True if line ends with sentence-terminating punctuation.

    Args:
        line: Stripped text line.

    Returns:
        True if the line ends with ., !, ?, or closing quote/apostrophe.
    """
    return bool(line) and line[-1] in _SENTENCE_ENDINGS


def _merge_lines(text: str, short_line_threshold: int) -> str:
    """Merge OCR-fragmented lines within a single block's text.

    Lines shorter than ``short_line_threshold`` that do not end a sentence
    are joined with the next line using a space. Lines that end a sentence
    or exceed the threshold are kept as paragraph breaks.

    Args:
        text: Raw block text (may contain embedded newlines).
        short_line_threshold: Lines shorter than this are candidates for merging.

    Returns:
        Text with fragmented lines rejoined into proper sentences.
    """
    lines = text.splitlines()
    if len(lines) <= 1:
        return text

    merged: list[str] = []
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if buffer:
                merged.append(" ".join(buffer))
                buffer = []
            merged.append("")
            continue

        buffer.append(stripped)

        # Flush buffer if this line ends a sentence or is long enough to be complete
        if _ends_sentence(stripped) or len(stripped) >= short_line_threshold:
            merged.append(" ".join(buffer))
            buffer = []

    if buffer:
        merged.append(" ".join(buffer))

    return "\n".join(line for line in merged if line or not merged)


class OcrLineMerger(BaseCleaner):
    """Cleaner that merges OCR-fragmented lines into complete paragraphs.

    OCR engines often split a single sentence across multiple short lines
    due to column detection or line-height ambiguity. This cleaner joins
    consecutive short lines that do not end a sentence into a single line,
    reconstructing the original paragraph flow.

    This cleaner is a no-op on non-OCR content — it only joins lines that
    are short and do not end with sentence-terminating punctuation.

    Usage::

        cleaner = OcrLineMerger()
        cleaned_blocks = cleaner.clean(blocks)

        # More aggressive merging
        cleaner = OcrLineMerger(short_line_threshold=120)

    Args:
        short_line_threshold: Lines shorter than this character count are
            treated as OCR fragments and merged with the next line.
            Defaults to 80.
    """

    def __init__(self, short_line_threshold: int = _DEFAULT_SHORT_LINE_THRESHOLD) -> None:
        self._threshold = short_line_threshold

    def clean(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Merge fragmented OCR lines within each block.

        Args:
            blocks: Input IRBlocks from the previous pipeline stage.

        Returns:
            List of IRBlocks with merged text. Block count is unchanged;
            only text content within each block is modified.
        """
        result: list[IRBlock] = []
        for block in blocks:
            merged = _merge_lines(block.text, self._threshold)
            if merged == block.text:
                result.append(block)
            else:
                result.append(block.model_copy(update={"text": merged}))
        return result
