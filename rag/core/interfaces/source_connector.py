"""Source connector interface — Task 15.1.

Defines the abstract base class that every external-source connector
(Email, Slack, Notion, Google Docs, …) must implement.

A connector is responsible for:
1. Fetching new or updated items since a saved cursor position.
2. Returning each item as a ``SourceArtifact``.
3. Advancing and exposing the cursor so the caller can persist it.
4. Reporting its own health via ``healthcheck()``.

Usage::

    class MyConnector(BaseSourceConnector):
        def list_items(self, since_cursor: str = "") -> list[SourceArtifact]: ...
        def next_cursor(self) -> str: ...
        def healthcheck(self) -> dict: ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.core.contracts.source_artifact import SourceArtifact


class BaseSourceConnector(ABC):
    """Abstract base class for all external source connectors.

    Implementations must be stateless between ``list_items`` calls except
    for the cursor: each call to ``list_items`` updates the internal cursor
    so that ``next_cursor()`` always returns the position *after* the last
    returned batch.

    Attributes:
        connector_name: Stable identifier used as the primary key in the
            ``connector_state`` table.  Must be unique across all connectors
            registered in ``configs/connectors/sources.yaml``.
    """

    #: Override in subclasses with a stable, lower-snake-case identifier.
    connector_name: str = ""

    @abstractmethod
    def list_items(self, since_cursor: str = "") -> list[SourceArtifact]:
        """Fetch items that are new or updated since ``since_cursor``.

        The cursor format is provider-specific (e.g. ISO timestamp, message
        ID, page token).  An empty string means "fetch from the beginning".

        Args:
            since_cursor: Opaque string cursor from a previous ``next_cursor()``
                call, or ``""`` to start from the beginning.

        Returns:
            Ordered list of ``SourceArtifact`` objects ready for ingestion.
            May be empty if there are no new items.
        """

    @abstractmethod
    def next_cursor(self) -> str:
        """Return the cursor position after the most recent ``list_items`` call.

        The returned value should be stored in the ``connector_state`` table
        via ``DocStore.save_connector_cursor()`` so subsequent runs can pass
        it back to ``list_items``.

        Returns:
            Opaque cursor string.  Empty string if no items have been fetched
            yet or if the provider does not support incremental sync.
        """

    @abstractmethod
    def healthcheck(self) -> dict:
        """Probe the remote service and return a status dict.

        Returns:
            Dictionary with at least the following keys:

            - ``"status"``: ``"ok"`` | ``"degraded"`` | ``"error"``
            - ``"connector"``: value of ``connector_name``
            - ``"detail"``: human-readable message (empty string on success)

        The implementation must never raise — errors should be captured and
        returned in the dict.
        """
