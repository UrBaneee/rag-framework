"""Plugin registry — factory functions for pluggable pipeline components.

Each factory reads the active configuration and returns the appropriate
implementation. New plugin types are added here as the framework grows.
"""

import logging
from typing import Any

from rag.core.interfaces.reranker import BaseReranker

logger = logging.getLogger(__name__)

# ── Reranker registry ──────────────────────────────────────────────────────────

_RERANKER_REGISTRY: dict[str, type[BaseReranker]] = {}


def register_reranker(name: str, cls: type[BaseReranker]) -> None:
    """Register a reranker implementation under a short name.

    Args:
        name: Short identifier used in configuration, e.g. ``"noop"`` or
            ``"voyage"``.
        cls: Reranker class (must implement ``BaseReranker``).
    """
    _RERANKER_REGISTRY[name] = cls
    logger.debug("Registered reranker '%s' → %s.", name, cls.__name__)


def build_reranker(config: dict[str, Any]) -> BaseReranker:
    """Instantiate the reranker specified in the configuration dict.

    Reads ``config["reranker"]["provider"]`` to select the implementation.
    Any additional keys under ``config["reranker"]`` are passed as keyword
    arguments to the reranker constructor.

    Falls back to the ``"noop"`` reranker if the provider key is absent or
    if the specified provider is not registered.

    Args:
        config: Configuration dict, typically loaded via
            ``rag.core.registry.config_loader.load_config``.

    Returns:
        An initialised ``BaseReranker`` instance.

    Example config::

        reranker:
          provider: noop

        reranker:
          provider: voyage
          model: rerank-2
          top_k: 8
    """
    # Lazy-register built-ins so they are always available
    _ensure_builtins_registered()

    reranker_cfg: dict[str, Any] = config.get("reranker", {})
    provider: str = reranker_cfg.get("provider", "noop")

    if provider not in _RERANKER_REGISTRY:
        logger.warning(
            "Unknown reranker provider '%s'. Falling back to 'noop'.", provider
        )
        provider = "noop"

    cls = _RERANKER_REGISTRY[provider]
    # Pass all config keys except 'provider' as kwargs
    kwargs = {k: v for k, v in reranker_cfg.items() if k != "provider"}

    try:
        instance = cls(**kwargs)
    except TypeError as exc:
        logger.warning(
            "Failed to instantiate reranker '%s' with kwargs %s: %s. "
            "Falling back to 'noop'.",
            provider,
            kwargs,
            exc,
        )
        instance = _RERANKER_REGISTRY["noop"]()

    logger.info("Built reranker: %s (provider=%s).", type(instance).__name__, provider)
    return instance


def _ensure_builtins_registered() -> None:
    """Register built-in rerankers if not already present."""
    if "noop" not in _RERANKER_REGISTRY:
        from rag.infra.rerank.noop import NoOpReranker
        register_reranker("noop", NoOpReranker)
