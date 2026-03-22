"""Abstract base class for vector index plugins."""

from abc import ABC, abstractmethod

from rag.core.contracts.candidate import Candidate
from rag.core.contracts.chunk import Chunk


class BaseVectorIndex(ABC):
    """Interface that all vector index plugins must implement.

    Vector indexes store dense embeddings and support approximate nearest
    neighbour search. Implementations include FAISS, Milvus, and Qdrant.

    The index is populated during ingestion (``add``) and queried during
    retrieval (``search``). Implementations must support persistence so
    that the index survives process restarts.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks (with pre-computed embeddings) to the index.

        Args:
            chunks: Chunks whose ``embedding`` field has been populated.

        Raises:
            ValueError: If any chunk has a None embedding.
        """

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[Candidate]:
        """Retrieve the top-k most similar chunks by vector similarity.

        Args:
            query_vector: Dense query embedding of length ``dim``.
            top_k: Number of candidates to return.

        Returns:
            List of Candidates ordered by descending vector similarity,
            with ``vector_score`` populated and ``retrieval_source`` set
            to ``RetrievalSource.VECTOR``.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to disk.

        Args:
            path: Directory path where index files are written.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously persisted index from disk.

        Args:
            path: Directory path containing the saved index files.
        """
