"""Tiny YAML config loader shared by every module.

Nothing fancy: read a YAML file into a dict, cache it so repeated
`load("patrol_config.yaml")` calls in different modules don't re-hit
disk, and give a clear error naming the file when a key is missing
instead of a bare KeyError three layers of dict-access deep.

Every module that needs config calls `config_loader.load(paths.X_CONFIG)`
rather than opening YAML itself, so tests can monkeypatch or clear the
cache (`config_loader.clear_cache()`) between cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_cache: dict[Path, dict[str, Any]] = {}


class ConfigError(RuntimeError):
    pass


def load(path: str | Path, required: bool = True) -> dict[str, Any]:
    """Load a YAML config file, caching by resolved path."""
    p = Path(path).resolve()
    if p in _cache:
        return _cache[p]

    if not p.exists():
        if required:
            raise ConfigError(f"missing config file: {p}")
        _cache[p] = {}
        return _cache[p]

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ConfigError(f"config file {p} must contain a YAML mapping at top level")

    _cache[p] = data
    return data


def get(config: dict[str, Any], dotted_key: str, default: Any = ..., *, source: str = "") -> Any:
    """Fetch a nested key using dot notation, e.g. get(cfg, 'zoom.max_x')."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is not ...:
                return default
            raise ConfigError(f"missing key '{dotted_key}' in {source or 'config'}")
        node = node[part]
    return node


def clear_cache() -> None:
    _cache.clear()
