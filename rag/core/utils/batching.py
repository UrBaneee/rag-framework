"""Batching utilities for splitting large sequences into fixed-size chunks."""

from collections.abc import Generator
from dataclasses import dataclass, field
from typing import TypeVar

T = TypeVar("T")


def iter_batches(items: list[T], batch_size: int) -> Generator[list[T], None, None]:
    """Yield successive fixed-size batches from a list.

    Args:
        items: The source list to split into batches.
        batch_size: Maximum number of items per batch. Must be >= 1.

    Yields:
        Successive sub-lists of length at most ``batch_size``.

    Raises:
        ValueError: If ``batch_size`` is less than 1.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@dataclass
class EmbedBatchAccumulator:
    """Aggregates vectors and token counts across multiple embedding batches.

    Usage::

        acc = EmbedBatchAccumulator()
        for batch in iter_batches(texts, batch_size=64):
            vectors, tokens = call_api(batch)
            acc.add(vectors, tokens)
        result = acc.vectors  # all vectors in order

    Attributes:
        vectors: Accumulated embedding vectors in input order.
        total_tokens: Sum of prompt tokens across all batches.
    """

    vectors: list[list[float]] = field(default_factory=list)
    total_tokens: int = 0

    def add(self, batch_vectors: list[list[float]], prompt_tokens: int = 0) -> None:
        """Append a batch of vectors and accumulate token usage.

        Args:
            batch_vectors: Vectors from one API call, in index order.
            prompt_tokens: Token count reported for this batch.
        """
        self.vectors.extend(batch_vectors)
        self.total_tokens += prompt_tokens
