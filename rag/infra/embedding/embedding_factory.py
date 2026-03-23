"""Embedding provider factory — instantiates the configured provider from settings."""

import importlib
import logging
from typing import Any

from rag.core.interfaces.embedding import BaseEmbeddingProvider

logger = logging.getLogger(__name__)

# Registry maps provider names (as used in settings.yaml) to
# "module.path:ClassName" strings for lazy import.
_PROVIDER_REGISTRY: dict[str, str] = {
    "openai": "rag.infra.embedding.openai_embedding:OpenAIEmbeddingProvider",
}


def register_provider(name: str, class_path: str) -> None:
    """Register a custom embedding provider.

    Args:
        name: Provider name as it will appear in settings.yaml.
        class_path: Dotted module path and class name separated by ``:``,
            e.g. ``"mypackage.mymodule:MyProvider"``.
    """
    _PROVIDER_REGISTRY[name] = class_path


def create_embedding_provider(config: dict[str, Any]) -> BaseEmbeddingProvider:
    """Instantiate an embedding provider from configuration.

    Reads ``config["embedding"]["provider"]`` to select the implementation,
    then passes the full ``config["embedding"]`` dict to the provider
    constructor as keyword arguments.

    Args:
        config: Top-level application config dict (as returned by
            ``load_config()``).

    Returns:
        An initialised ``BaseEmbeddingProvider`` instance.

    Raises:
        KeyError: If ``embedding.provider`` is missing from the config.
        ValueError: If the provider name is not in the registry.
        ImportError: If the provider module cannot be imported.
    """
    embedding_cfg: dict[str, Any] = config.get("embedding", {})
    provider_name: str = embedding_cfg.get("provider", "")

    if not provider_name:
        raise KeyError("config['embedding']['provider'] is required but not set.")

    if provider_name not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown embedding provider '{provider_name}'. "
            f"Available: {sorted(_PROVIDER_REGISTRY)}"
        )

    class_path = _PROVIDER_REGISTRY[provider_name]
    module_path, class_name = class_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Could not import embedding provider module '{module_path}': {exc}"
        ) from exc

    cls = getattr(module, class_name)
    provider_kwargs = {k: v for k, v in embedding_cfg.items() if k != "provider"}

    logger.debug(
        "Creating embedding provider '%s' with kwargs: %s",
        provider_name,
        list(provider_kwargs),
    )

    return cls(**provider_kwargs)
