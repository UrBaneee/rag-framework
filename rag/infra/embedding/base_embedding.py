"""Base embedding provider — re-exports the core ABC and defines EmbeddingResult."""

from dataclasses import dataclass, field

from rag.core.interfaces.embedding import BaseEmbeddingProvider

__all__ = ["BaseEmbeddingProvider", "EmbeddingResult"]


@dataclass
class EmbeddingResult:
    """Container returned by embedding providers.

    Attributes:
        vectors: List of dense embedding vectors, one per input text.
        model: Model identifier used to produce the embeddings.
        prompt_tokens: Number of tokens consumed by the input texts.
    """

    vectors: list[list[float]]
    model: str
    prompt_tokens: int = 0

    def __len__(self) -> int:
        return len(self.vectors)
