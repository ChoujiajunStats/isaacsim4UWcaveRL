"""External cave asset registry and path resolution.

Large cave meshes remain outside the repository.  This module resolves a
small YAML description into paths and keeps the provenance/status fields
available to task setup and validation scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .base import OpenWaterWorldCfg


@dataclass(frozen=True)
class CaveAssetPaths:
    name: str
    visual_source: Path
    collision_source: Path
    metadata_path: Path | None = None
    centerline_path: Path | None = None
    scale_status: str = "unknown"
    candidate_status: str = "unknown"

    def require_sources(self) -> None:
        missing = [str(path) for path in (self.visual_source, self.collision_source) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing cave asset source(s): " + ", ".join(missing))


@dataclass
class CaveWorldCfg(OpenWaterWorldCfg):
    """World configuration for a visual mesh plus independent collision mesh."""

    visual_asset_path: str | None = None
    collision_asset_path: str | None = None
    asset_name: str = "Cave"
    clone_per_env: bool = True
    centerline_path: str | None = None
    spawn_chainage_m: float = 2.0
    goal_chainage_m: float = 20.0
    spawn_jitter_m: float = 0.0
    visual_enabled: bool = True

    def __post_init__(self) -> None:
        if self.visual_asset_path and not self.asset_path:
            self.asset_path = self.visual_asset_path


def cave_asset_root() -> Path:
    return Path(os.environ.get("ISAAC_UNDERWATER_ASSET_ROOT", "/home/kenton/Downloads/assets")).expanduser().resolve()


def resolve_cave_asset(config_path: str | Path, *, require_sources: bool = True) -> CaveAssetPaths:
    """Resolve a project cave config without copying the external asset tree."""
    path = Path(config_path)
    if not path.is_absolute():
        path = path if path.is_file() else Path(__file__).resolve().parents[4] / "configs" / path
    with path.open(encoding="utf-8") as stream:
        data: dict[str, Any] = yaml.safe_load(stream) or {}
    root = Path(os.environ.get(str(data.get("asset_root_env", "ISAAC_UNDERWATER_ASSET_ROOT")), data.get("asset_root_default", cave_asset_root())))
    root = root.expanduser().resolve()

    def resolve(relative: str | None) -> Path | None:
        return None if not relative else (root / relative).resolve()

    result = CaveAssetPaths(
        name=str(data["name"]),
        visual_source=resolve(str(data["visual_source"])),
        collision_source=resolve(str(data["collision_source"])),
        metadata_path=resolve(data.get("metadata")),
        centerline_path=resolve(data.get("centerline")),
        scale_status=str(data.get("scale_status", "unknown")),
        candidate_status=str(data.get("candidate_status", "unknown")),
    )
    if require_sources:
        result.require_sources()
    return result


def load_cave_world_cfg(
    config_path: str | Path,
    usd_root: str | Path | None = None,
    *,
    require_converted: bool = True,
) -> CaveWorldCfg:
    """Build a simulator world config from a registry YAML and converted USDs."""
    path = Path(config_path)
    if not path.is_absolute():
        path = path if path.is_file() else Path(__file__).resolve().parents[4] / "configs" / path
    with path.open(encoding="utf-8") as stream:
        data: dict[str, Any] = yaml.safe_load(stream) or {}
    assets = resolve_cave_asset(path, require_sources=require_converted)
    output_root = Path(
        usd_root
        or os.environ.get(
            "ISAAC_UNDERWATER_CAVE_USD_ROOT",
            str(Path(__file__).resolve().parents[4] / "outputs" / "cave_assets"),
        )
    ).expanduser().resolve()
    name = assets.name
    visual_usd = output_root / name / "visual.usd"
    collision_usd = output_root / name / "collision.usd"
    if require_converted and (not visual_usd.is_file() or not collision_usd.is_file()):
        raise FileNotFoundError(
            f"Converted cave USDs are missing for {name}. Run scripts/convert_cave_asset.py "
            f"--config {path} first (expected {visual_usd} and {collision_usd})."
        )
    return CaveWorldCfg(
        visual_asset_path=str(visual_usd),
        collision_asset_path=str(collision_usd),
        asset_scale=float(data.get("asset_scale", 1.0)),
        asset_translation_m=tuple(float(x) for x in data.get("asset_translation_m", (0.0, 0.0, 0.0))),
        asset_orientation_wxyz=tuple(
            float(x) for x in data.get("asset_orientation_wxyz", (1.0, 0.0, 0.0, 0.0))
        ),
        asset_name="Cave",
        clone_per_env=bool(data.get("clone_per_env", True)),
        centerline_path=str(assets.centerline_path) if assets.centerline_path else None,
        spawn_chainage_m=float(data.get("spawn_chainage_m", 2.0)),
        goal_chainage_m=float(data.get("goal_chainage_m", 20.0)),
        spawn_jitter_m=float(data.get("spawn_jitter_m", 0.0)),
        navigation_workspace_size_m=tuple(
            float(x) for x in data.get("navigation_workspace_size_m", (110.0, 50.0, 8.0))
        ),
        visual_enabled=bool(data.get("visual_enabled", True)),
    )
