"""Config loader — loads settings.yaml and applies profile overrides."""

import copy
import os
from pathlib import Path
from typing import Any, Optional

import yaml


# ── Path resolution ────────────────────────────────────────────────────────────

def _project_root() -> Path:
    """Resolve the project root directory.

    Uses the ``RAG_PROJECT_ROOT`` environment variable if set, otherwise
    walks up from this file's location until a ``configs/`` directory is found.

    Returns:
        Absolute Path to the project root.

    Raises:
        FileNotFoundError: If no configs/ directory is found in the hierarchy.
    """
    env_root = os.environ.get("RAG_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # Walk up from this file until we find configs/
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "configs").is_dir():
            return parent

    raise FileNotFoundError(
        "Could not locate project root (no configs/ directory found). "
        "Set RAG_PROJECT_ROOT environment variable to fix this."
    )


# ── Deep merge ─────────────────────────────────────────────────────────────────

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict.

    For each key in override:
    - If both values are dicts, merge recursively.
    - Otherwise, override replaces base.

    Args:
        base: The base configuration dict.
        override: The profile or override dict to apply on top.

    Returns:
        A new merged dict. Neither input is modified.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def load_config(profile: Optional[str] = None) -> dict[str, Any]:
    """Load the RAG framework configuration.

    Loads ``configs/settings.yaml`` as the base configuration, then
    optionally applies a profile override from
    ``configs/profiles/<profile>.yaml``.

    Args:
        profile: Profile name to apply, e.g. "local_fast" or
            "local_quality". If None, returns the base config only.

    Returns:
        A single merged configuration dict.

    Raises:
        FileNotFoundError: If settings.yaml or the requested profile file
            does not exist.
        yaml.YAMLError: If any YAML file contains invalid syntax.
    """
    root = _project_root()

    settings_path = root / "configs" / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {settings_path}")

    with settings_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if profile is not None:
        profile_path = root / "configs" / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(
                f"Profile '{profile}' not found at {profile_path}"
            )
        with profile_path.open(encoding="utf-8") as f:
            profile_data = yaml.safe_load(f) or {}
        config = _deep_merge(config, profile_data)

    return config
