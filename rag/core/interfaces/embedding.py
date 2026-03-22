"""Abstract base class for embedding provider plugins."""

from abc import ABC, abstractmethod


class BaseEmbeddingProvider(ABC):
    """Interface that all embedding provider plugins must implement.

    Embedding providers convert text strings into dense float vectors.
    Implementations include OpenAI, Voyage, Cohere, and local Ollama models.

    All providers must produce vectors of a consistent dimensionality
    (reported via the ``dim`` property) so that the vector index can be
    initialised correctly.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider.

        Returns:
            Integer vector dimension, e.g. 1536 for text-embedding-3-small.
        """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into dense vectors.

        Args:
            texts: List of text strings to embed. Must not be empty.

        Returns:
            List of embedding vectors, one per input text, each of length
            ``self.dim``.

        Raises:
            ValueError: If ``texts`` is empty.
        """
