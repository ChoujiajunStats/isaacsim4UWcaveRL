"""Load project-owned YAML configuration without coupling it to simulator startup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def config_dir() -> Path:
    override = os.environ.get("ISAAC_UNDERWATER_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "configs"


def load_config(filename: str) -> dict[str, Any]:
    path = config_dir() / filename
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data
