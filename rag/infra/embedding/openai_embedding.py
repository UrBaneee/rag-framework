"""OpenAI embedding provider — wraps the OpenAI Embeddings API."""

import logging

try:
    from openai import OpenAI
except ImportError as _openai_import_error:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]
    _openai_import_error_msg = str(_openai_import_error)
else:
    _openai_import_error_msg = ""

from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.infra.embedding.base_embedding import EmbeddingResult

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by the OpenAI Embeddings API.

    Sends text batches to ``client.embeddings.create`` and returns dense
    float vectors alongside token usage statistics.

    Usage::

        provider = OpenAIEmbeddingProvider(model="text-embedding-3-small", dim=1536)
        vectors = provider.embed(["Hello world", "Goodbye world"])

    Args:
        model: OpenAI embedding model identifier.
        dim: Expected vector dimensionality for the chosen model.
        api_key: OpenAI API key. If None, the ``OPENAI_API_KEY`` environment
            variable is used (standard OpenAI client behaviour).
        batch_size: Maximum number of texts per API call. Defaults to 64.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        api_key: str | None = None,
        batch_size: int = 64,
        **kwargs,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingProvider. "
                f"Install it with: pip install openai ({_openai_import_error_msg})"
            )

        self._model = model
        self._dim = dim
        self._batch_size = batch_size
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this provider."""
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the OpenAI Embeddings API.

        Args:
            texts: Non-empty list of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If ``texts`` is empty.
            openai.OpenAIError: On API-level errors.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        all_vectors: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
            except Exception as exc:
                logger.error(
                    "OpenAI embeddings API call failed (batch %d–%d): %s",
                    i,
                    i + len(batch),
                    exc,
                )
                raise

            # Response items are ordered by index field, not position
            sorted_data = sorted(response.data, key=lambda item: item.index)
            all_vectors.extend(item.embedding for item in sorted_data)

            logger.debug(
                "Embedded batch %d–%d, tokens used: %d",
                i,
                i + len(batch),
                response.usage.prompt_tokens,
            )

        return all_vectors

    def embed_with_usage(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts and return vectors plus token usage.

        Args:
            texts: Non-empty list of text strings to embed.

        Returns:
            EmbeddingResult with vectors, model name, and prompt token count.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        all_vectors: list[list[float]] = []
        total_tokens = 0

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )
            except Exception as exc:
                logger.error("OpenAI embeddings API call failed: %s", exc)
                raise

            sorted_data = sorted(response.data, key=lambda item: item.index)
            all_vectors.extend(item.embedding for item in sorted_data)
            total_tokens += response.usage.prompt_tokens

        return EmbeddingResult(
            vectors=all_vectors,
            model=self._model,
            prompt_tokens=total_tokens,
        )
