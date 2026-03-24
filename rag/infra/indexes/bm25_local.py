"""Local BM25 keyword index backed by rank-bm25."""

import logging
import pickle
import re
from pathlib import Path

from rag.core.contracts.candidate import Candidate, RetrievalSource
from rag.core.contracts.chunk import Chunk
from rag.core.interfaces.keyword_index import BaseKeywordIndex

logger = logging.getLogger(__name__)

_INDEX_FILENAME = "bm25.pkl"

# Jieba Chinese word segmenter — optional but required for CJK BM25 to work.
# Without it, Chinese text is treated as a single token and BM25 scores 0.
try:
    import jieba as _jieba
    _jieba.setLogLevel(logging.WARNING)  # suppress jieba's verbose init logs
    _JIEBA_AVAILABLE = True
except ImportError:
    _JIEBA_AVAILABLE = False

# Matches any CJK Unified Ideograph (core Chinese characters)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def _has_cjk(text: str) -> bool:
    """Return True if *text* contains at least one CJK character."""
    return bool(_CJK_RE.search(text))


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 indexing and querying.

    Strategy:
    - CJK text  -> jieba word segmentation (e.g. "帆海资本" -> ["帆海", "资本"])
                   Falls back to single-character tokens when jieba is absent.
    - Latin text -> lowercase alphanumeric split (unchanged behaviour).

    Args:
        text: Raw text string.

    Returns:
        List of tokens suitable for BM25 scoring.
    """
    if _has_cjk(text):
        if _JIEBA_AVAILABLE:
            return [t for t in _jieba.cut(text) if t.strip()]
        else:
            logger.warning(
                "jieba not installed -- CJK text tokenized as characters. "
                "Install with: pip install jieba"
            )
            return [ch for ch in text if ch.strip()]
    return re.findall(r"\w+", text.lower())


class BM25LocalIndex(BaseKeywordIndex):
    """BM25 keyword index stored entirely in memory with optional disk persistence.

    Uses ``rank_bm25.BM25Okapi`` for scoring. The index is built over the
    ``stable_text`` field of each chunk (not ``display_text``).

    Chunk removal triggers a full index rebuild, which is acceptable for the
    expected corpus sizes (thousands of chunks).

    Usage::

        index = BM25LocalIndex()
        index.add(chunks)
        candidates = index.search("query terms", top_k=10)
        index.save("/path/to/dir")

        index2 = BM25LocalIndex()
        index2.load("/path/to/dir")
        candidates2 = index2.search("query terms", top_k=10)
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None  # rank_bm25.BM25Okapi instance, built lazily

    # ── Internal ──────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        """Rebuild the BM25 model from current chunk list."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for BM25LocalIndex. "
                "Install with: pip install rank-bm25"
            ) from exc

        if not self._chunks:
            self._bm25 = None
            return

        tokenized = [_tokenize(c.stable_text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("BM25 index rebuilt with %d chunks.", len(self._chunks))

    # ── BaseKeywordIndex ──────────────────────────────────────────────────────

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the index.

        Args:
            chunks: Chunks to add. Existing chunks are not deduplicated here;
                callers should avoid adding the same chunk_id twice.
        """
        self._chunks.extend(chunks)
        self._rebuild()

    def search(self, query: str, top_k: int) -> list[Candidate]:
        """Retrieve the top-k most relevant chunks by BM25 score.

        Args:
            query: Raw query string.
            top_k: Maximum number of candidates to return.

        Returns:
            Candidates ordered by descending BM25 score.
            Returns an empty list if the index is empty.
        """
        if self._bm25 is None or not self._chunks:
            return []

        tokens = _tokenize(query)
        scores: list[float] = self._bm25.get_scores(tokens).tolist()

        # Pair each score with its chunk, sort descending, take top_k
        scored = sorted(
            zip(scores, self._chunks), key=lambda x: x[0], reverse=True
        )[:top_k]

        candidates = []
        for score, chunk in scored:
            candidates.append(
                Candidate(
                    chunk_id=chunk.chunk_id or "",
                    doc_id=chunk.doc_id,
                    display_text=chunk.display_text,
                    stable_text=chunk.stable_text,
                    bm25_score=score,
                    retrieval_source=RetrievalSource.BM25,
                    metadata=chunk.metadata,
                )
            )
        return candidates

    def remove(self, chunk_id: str) -> None:
        """Remove a chunk from the index by its chunk_id.

        Triggers a full index rebuild.

        Args:
            chunk_id: The chunk_id of the chunk to remove.
        """
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.chunk_id != chunk_id]
        if len(self._chunks) < before:
            self._rebuild()
            logger.debug("Removed chunk '%s' from BM25 index.", chunk_id)
        else:
            logger.warning("chunk_id '%s' not found in BM25 index.", chunk_id)

    def save(self, path: str) -> None:
        """Persist the index state to disk.

        Writes ``bm25.pkl`` containing the chunk list and BM25 model.

        Args:
            path: Directory path where the index file is written.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        index_path = dir_path / _INDEX_FILENAME

        payload = {"chunks": self._chunks, "bm25": self._bm25}
        try:
            with index_path.open("wb") as f:
                pickle.dump(payload, f)
        except OSError as exc:
            raise OSError(f"Failed to save BM25 index to '{index_path}': {exc}") from exc

        logger.debug("BM25 index saved to '%s'.", index_path)

    def load(self, path: str) -> None:
        """Load a previously saved index from disk.

        Args:
            path: Directory path containing ``bm25.pkl``.

        Raises:
            FileNotFoundError: If the index file does not exist.
        """
        index_path = Path(path) / _INDEX_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 index file not found at '{index_path}'. "
                "Run add() and save() first."
            )

        try:
            with index_path.open("rb") as f:
                payload = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as exc:
            raise OSError(f"Failed to load BM25 index from '{index_path}': {exc}") from exc

        self._chunks = payload["chunks"]
        self._bm25 = payload["bm25"]
        logger.debug("BM25 index loaded from '%s' (%d chunks).", index_path, len(self._chunks))
