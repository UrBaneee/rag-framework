"""Multilingual embedding provider — wraps sentence-transformers for cross-lingual search.

Uses local models from the sentence-transformers library, which run entirely
on-device (no API key needed). The default model ``paraphrase-multilingual-mpnet-base-v2``
supports 50+ languages and maps semantically equivalent text in different
scripts to nearby vectors, enabling Chinese queries to retrieve English
documents and vice versa.

Recommended models:
  - paraphrase-multilingual-mpnet-base-v2  (768-dim, 50+ langs, good quality)
  - paraphrase-multilingual-MiniLM-L12-v2  (384-dim, faster, slightly lower quality)
  - intfloat/multilingual-e5-large         (1024-dim, state-of-the-art, slower)
  - intfloat/multilingual-e5-base          (768-dim, good balance)
"""

import logging

from rag.core.interfaces.embedding import BaseEmbeddingProvider
from rag.infra.embedding.base_embedding import EmbeddingResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
_DEFAULT_BATCH_SIZE = 32


class MultilingualEmbeddingProvider(BaseEmbeddingProvider):
    """Local multilingual embedding provider via sentence-transformers.

    Encodes texts into dense vectors using a multilingual transformer model
    that has been trained to place semantically equivalent sentences from
    different languages near each other in the embedding space.

    This enables true cross-lingual retrieval: a Chinese query can find
    relevant English documents and vice versa, without any translation step.

    Usage::

        provider = MultilingualEmbeddingProvider()
        vectors = provider.embed(["你好世界", "Hello world"])
        # Both vectors will be close in cosine space.

    Args:
        model: sentence-transformers model name or HuggingFace model ID.
            Defaults to ``paraphrase-multilingual-mpnet-base-v2``.
        dim: Expected vector dimensionality. If None, inferred automatically
            from the model on first embed call.
        batch_size: Texts per encoding batch. Larger = faster but more memory.
        device: PyTorch device string (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).
            If None, sentence-transformers auto-selects.
        normalize: If True, L2-normalise output vectors (cosine similarity
            becomes equivalent to dot product). Defaults to True.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        dim: int | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device: str | None = None,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for MultilingualEmbeddingProvider. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._model_name = model
        self._batch_size = batch_size
        self._normalize = normalize

        logger.info("Loading multilingual model '%s' …", model)
        load_kwargs: dict = {}
        if device:
            load_kwargs["device"] = device
        self._encoder = SentenceTransformer(model, **load_kwargs)

        # Resolve dim — override if explicitly provided
        if dim is not None:
            self._dim = dim
        else:
            self._dim = self._encoder.get_sentence_embedding_dimension()  # type: ignore[assignment]
        logger.info(
            "Multilingual model ready: %s  dim=%d", model, self._dim
        )

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into dense multilingual vectors.

        Args:
            texts: Non-empty list of text strings (any language mix).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        return self.embed_with_usage(texts).vectors

    def embed_with_usage(self, texts: list[str]) -> EmbeddingResult:
        """Embed texts and return vectors plus token usage estimate.

        sentence-transformers does not report token counts, so ``prompt_tokens``
        is estimated as the sum of whitespace-split token counts.

        Args:
            texts: Non-empty list of text strings.

        Returns:
            EmbeddingResult with vectors, model name, and estimated token count.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        embeddings = self._encoder.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        vectors = [vec.tolist() for vec in embeddings]

        # Estimate token count (sentence-transformers does not expose usage)
        estimated_tokens = sum(len(t.split()) for t in texts)

        logger.debug(
            "Multilingual embed: %d text(s), estimated %d tokens",
            len(texts),
            estimated_tokens,
        )

        return EmbeddingResult(
            vectors=vectors,
            model=self._model_name,
            prompt_tokens=estimated_tokens,
        )
