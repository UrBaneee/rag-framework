"""Index startup loader — initialises or reloads BM25 and FAISS indexes from disk."""

import logging
from pathlib import Path

from rag.infra.indexes.bm25_local import BM25LocalIndex, _INDEX_FILENAME as _BM25_FILE
from rag.infra.indexes.faiss_local import FaissLocalIndex, _INDEX_FILENAME as _FAISS_FILE

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages lifecycle of BM25 and FAISS indexes with disk persistence.

    On construction, ``IndexManager`` checks whether saved index files exist
    at ``index_dir``:

    - If both ``faiss.index`` / ``faiss_mapping.pkl`` and ``bm25.pkl`` are
      found, they are loaded into memory immediately.
    - If the directory is absent or the files do not exist, fresh empty
      indexes are created (first-run case).

    After ingestion the caller should call :meth:`save` to persist the
    updated indexes so they survive process restarts.

    Usage::

        manager = IndexManager("/data/indexes")
        # manager.bm25 and manager.faiss are ready to use

        # After ingestion:
        manager.save()

    Args:
        index_dir: Directory where index files are stored.
    """

    def __init__(self, index_dir: str | Path) -> None:
        self._index_dir = Path(index_dir)
        self.bm25: BM25LocalIndex = BM25LocalIndex()
        self.faiss: FaissLocalIndex = FaissLocalIndex()
        self._load_if_exists()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        """Load indexes from disk if saved files are present."""
        bm25_path = self._index_dir / _BM25_FILE
        faiss_path = self._index_dir / _FAISS_FILE

        if bm25_path.exists():
            try:
                self.bm25.load(str(self._index_dir))
                logger.info("BM25 index loaded from '%s'.", self._index_dir)
            except Exception as exc:
                logger.warning(
                    "Failed to load BM25 index from '%s': %s. Starting empty.",
                    self._index_dir,
                    exc,
                )
                self.bm25 = BM25LocalIndex()
        else:
            logger.info(
                "No BM25 index found at '%s'. Starting with empty index.", self._index_dir
            )

        if faiss_path.exists():
            try:
                self.faiss.load(str(self._index_dir))
                logger.info("FAISS index loaded from '%s'.", self._index_dir)
            except Exception as exc:
                logger.warning(
                    "Failed to load FAISS index from '%s': %s. Starting empty.",
                    self._index_dir,
                    exc,
                )
                self.faiss = FaissLocalIndex()
        else:
            logger.info(
                "No FAISS index found at '%s'. Starting with empty index.", self._index_dir
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist both indexes to disk.

        Creates ``index_dir`` if it does not yet exist.
        """
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self.bm25.save(str(self._index_dir))
        self.faiss.save(str(self._index_dir))
        logger.info("BM25 and FAISS indexes saved to '%s'.", self._index_dir)

    def reload(self) -> None:
        """Reload both indexes from disk, replacing current in-memory state.

        Useful after an external process has updated the index files.
        """
        self.bm25 = BM25LocalIndex()
        self.faiss = FaissLocalIndex()
        self._load_if_exists()
        logger.info("Indexes reloaded from '%s'.", self._index_dir)
