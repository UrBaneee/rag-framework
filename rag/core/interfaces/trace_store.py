"""Abstract base class for pipeline trace store plugins."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from rag.core.contracts.trace import AnswerTrace


class BaseTraceStore(ABC):
    """Interface that all trace store plugins must implement.

    The TraceStore persists pipeline run metadata and per-step trace events
    for observability. It supports the Streamlit Studio debug views.

    Core tables (from Section 14):
        runs, trace_events
    """

    @abstractmethod
    def save_run(self, run_type: str, metadata: dict[str, Any]) -> str:
        """Record a new pipeline run and return its run_id.

        Args:
            run_type: Pipeline type identifier, e.g. "ingest" or "query".
            metadata: Arbitrary metadata about this run (query, source path,
                config snapshot, etc.).

        Returns:
            A unique run_id string for this run.
        """

    @abstractmethod
    def save_answer_trace(self, run_id: str, trace: AnswerTrace) -> None:
        """Persist a complete AnswerTrace for a query run.

        Args:
            run_id: Run identifier returned by ``save_run``.
            trace: The AnswerTrace to store.
        """

    @abstractmethod
    def get_answer_trace(self, run_id: str) -> Optional[AnswerTrace]:
        """Retrieve an AnswerTrace by run_id.

        Args:
            run_id: Run identifier.

        Returns:
            The AnswerTrace if found, or None.
        """

    @abstractmethod
    def list_runs(self, run_type: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
        """List recent pipeline runs.

        Args:
            run_type: Filter by run type ("ingest" or "query"), or None for all.
            limit: Maximum number of runs to return.

        Returns:
            List of run metadata dicts ordered by most recent first.
        """
