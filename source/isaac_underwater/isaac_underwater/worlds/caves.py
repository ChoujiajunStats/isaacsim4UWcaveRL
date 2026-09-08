"""External cave asset registry and path resolution.

Large cave meshes remain outside the repository.  This module resolves a
small YAML description into paths and keeps the provenance/status fields
available to task setup and validation scripts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

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
    key: str = "default"
    difficulty: str = "unknown"
    navigation_path: Path | None = None
    collision_status: str = "unknown"

    def require_sources(self) -> None:
        missing = [str(path) for path in (self.visual_source, self.collision_source) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing cave asset source(s): " + ", ".join(missing))


@dataclass(frozen=True)
class CaveSceneCfg:
    """One statically assigned cave variant in a vectorized scene."""

    key: str
    name: str
    difficulty: str
    visual_asset_path: str
    collision_asset_path: str
    navigation_path: str
    metadata_path: str | None = None
    asset_scale: float = 1.0
    asset_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    asset_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    spawn_chainage_m: float | None = None
    goal_chainage_m: float | None = None
    spawn_jitter_m: float = 0.0
    collision_status: str = "unknown"


@dataclass(frozen=True)
class CaveDatasetPaths:
    """Resolved external sources selected by a named dataset profile."""

    name: str
    profile: str
    scenes: tuple[CaveAssetPaths, ...]
    navigation_workspace_size_m: tuple[float, float, float]
    checksum_manifest_path: Path | None = None
    license_status: str = "unknown"


def coerce_cave_scene_cfg(value: CaveSceneCfg | Mapping[str, Any]) -> CaveSceneCfg:
    """Restore nested scene dataclasses after Hydra converts them to dicts."""
    if isinstance(value, CaveSceneCfg):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected CaveSceneCfg or mapping, got {type(value).__name__}")
    data = dict(value)
    for field in ("asset_translation_m", "asset_orientation_wxyz"):
        if field in data:
            data[field] = tuple(float(component) for component in data[field])
    return CaveSceneCfg(**data)


@dataclass
class CaveWorldCfg(OpenWaterWorldCfg):
    """World configuration for a visual mesh plus independent collision mesh."""

    visual_asset_path: str | None = None
    seabed_enabled: bool = False
    collision_asset_path: str | None = None
    asset_name: str = "Cave"
    clone_per_env: bool = True
    centerline_path: str | None = None
    spawn_chainage_m: float = 2.0
    goal_chainage_m: float = 20.0
    spawn_jitter_m: float = 0.0
    visual_enabled: bool = True
    dataset_name: str | None = None
    dataset_profile: str | None = None
    scene_variants: tuple[CaveSceneCfg, ...] = ()

    def __post_init__(self) -> None:
        if self.visual_asset_path and not self.asset_path:
            self.asset_path = self.visual_asset_path


def cave_asset_root() -> Path:
    return Path(os.environ.get("ISAAC_UNDERWATER_ASSET_ROOT", "/home/kenton/Downloads/assets")).expanduser().resolve()


def _project_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if not path.is_absolute():
        path = path if path.is_file() else Path(__file__).resolve().parents[4] / "configs" / path
    return path.expanduser().resolve()


def _configured_asset_root(data: dict[str, Any]) -> Path:
    environment_name = str(data.get("asset_root_env", "ISAAC_UNDERWATER_ASSET_ROOT"))
    default = data.get("asset_root_default", cave_asset_root())
    return Path(os.environ.get(environment_name, default)).expanduser().resolve()


def resolve_cave_asset(config_path: str | Path, *, require_sources: bool = True) -> CaveAssetPaths:
    """Resolve a project cave config without copying the external asset tree."""
    path = _project_config_path(config_path)
    with path.open(encoding="utf-8") as stream:
        data: dict[str, Any] = yaml.safe_load(stream) or {}
    root = _configured_asset_root(data)

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


def resolve_cave_dataset(
    config_path: str | Path,
    profile: str = "train_all",
    *,
    require_sources: bool = True,
) -> CaveDatasetPaths:
    """Resolve a named multi-cave profile without copying external meshes."""
    path = _project_config_path(config_path)
    with path.open(encoding="utf-8") as stream:
        data: dict[str, Any] = yaml.safe_load(stream) or {}
    raw_scenes = data.get("scenes")
    raw_profiles = data.get("profiles")
    if not isinstance(raw_scenes, dict) or not raw_scenes:
        raise ValueError(f"Cave dataset has no scene mapping: {path}")
    if not isinstance(raw_profiles, dict) or profile not in raw_profiles:
        available = ", ".join(sorted(raw_profiles)) if isinstance(raw_profiles, dict) else "none"
        raise ValueError(f"Unknown cave dataset profile {profile!r}; available: {available}")
    selected = raw_profiles[profile]
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"Cave dataset profile {profile!r} must select at least one scene")
    if len(set(str(key) for key in selected)) != len(selected):
        raise ValueError(f"Cave dataset profile {profile!r} contains duplicate scene keys")

    root = _configured_asset_root(data)

    def resolve(relative: str | None) -> Path | None:
        return None if not relative else (root / relative).resolve()

    scenes: list[CaveAssetPaths] = []
    for raw_key in selected:
        key = str(raw_key)
        if key not in raw_scenes or not isinstance(raw_scenes[key], dict):
            raise ValueError(f"Profile {profile!r} references unknown scene {key!r}")
        scene: dict[str, Any] = raw_scenes[key]
        visual_source = resolve(str(scene["visual_source"]))
        collision_source = resolve(str(scene.get("collision_source", scene["visual_source"])))
        assert visual_source is not None and collision_source is not None
        asset = CaveAssetPaths(
            name=str(scene.get("name", f"{data.get('name', 'caves')}_{key}")),
            visual_source=visual_source,
            collision_source=collision_source,
            metadata_path=resolve(scene.get("metadata")),
            centerline_path=resolve(scene.get("navigation")),
            scale_status=str(scene.get("scale_status", data.get("scale_status", "unknown"))),
            candidate_status=str(scene.get("candidate_status", data.get("candidate_status", "unknown"))),
            key=key,
            difficulty=str(scene.get("difficulty", key)),
            navigation_path=resolve(scene.get("navigation")),
            collision_status=str(scene.get("collision_status", "unknown")),
        )
        if require_sources:
            asset.require_sources()
            for label, required_path in (
                ("navigation", asset.navigation_path),
                ("metadata", asset.metadata_path),
            ):
                if required_path is not None and not required_path.is_file():
                    raise FileNotFoundError(f"Missing cave {label} file: {required_path}")
        scenes.append(asset)
    workspace = tuple(float(value) for value in data.get("navigation_workspace_size_m", (180.0, 120.0, 20.0)))
    if len(workspace) != 3 or any(value <= 0.0 for value in workspace):
        raise ValueError(f"navigation_workspace_size_m must contain three positive values: {path}")
    checksum_manifest_path = resolve(data.get("checksums"))
    if require_sources and checksum_manifest_path is not None and not checksum_manifest_path.is_file():
        raise FileNotFoundError(f"Missing cave checksum manifest: {checksum_manifest_path}")
    return CaveDatasetPaths(
        name=str(data["name"]),
        profile=profile,
        scenes=tuple(scenes),
        navigation_workspace_size_m=workspace,  # type: ignore[arg-type]
        checksum_manifest_path=checksum_manifest_path,
        license_status=str(data.get("license_status", "unknown")),
    )


def load_cave_world_cfg(
    config_path: str | Path,
    usd_root: str | Path | None = None,
    *,
    require_converted: bool = True,
) -> CaveWorldCfg:
    """Build a simulator world config from a registry YAML and converted USDs."""
    path = _project_config_path(config_path)
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


def load_cave_dataset_world_cfg(
    config_path: str | Path,
    profile: str = "train_all",
    usd_root: str | Path | None = None,
    *,
    require_converted: bool = True,
) -> CaveWorldCfg:
    """Build a heterogeneous cave-world config for balanced vector training."""
    path = _project_config_path(config_path)
    with path.open(encoding="utf-8") as stream:
        data: dict[str, Any] = yaml.safe_load(stream) or {}
    dataset = resolve_cave_dataset(path, profile, require_sources=require_converted)
    output_root = Path(
        usd_root
        or os.environ.get(
            "ISAAC_UNDERWATER_CAVE_USD_ROOT",
            str(Path(__file__).resolve().parents[4] / "outputs" / "cave_assets"),
        )
    ).expanduser().resolve()
    raw_scenes: dict[str, dict[str, Any]] = data["scenes"]
    variants: list[CaveSceneCfg] = []
    for asset in dataset.scenes:
        raw = raw_scenes[asset.key]
        visual_usd = output_root / asset.name / "visual.usd"
        collision_usd = output_root / asset.name / "collision.usd"
        if require_converted and (not visual_usd.is_file() or not collision_usd.is_file()):
            raise FileNotFoundError(
                f"Converted cave USDs are missing for {asset.name}. Run scripts/convert_cave_asset.py "
                f"--config {path} --profile {profile} first."
            )
        if asset.navigation_path is None:
            raise ValueError(f"Dataset scene {asset.key!r} has no navigation path")
        variants.append(
            CaveSceneCfg(
                key=asset.key,
                name=asset.name,
                difficulty=asset.difficulty,
                visual_asset_path=str(visual_usd),
                collision_asset_path=str(collision_usd),
                navigation_path=str(asset.navigation_path),
                metadata_path=str(asset.metadata_path) if asset.metadata_path else None,
                asset_scale=float(raw.get("asset_scale", 1.0)),
                asset_translation_m=tuple(
                    float(value)
                    for value in raw.get("asset_translation_m", (0.0, 0.0, 0.0))
                ),
                asset_orientation_wxyz=tuple(
                    float(value)
                    for value in raw.get(
                        "asset_orientation_wxyz", (1.0, 0.0, 0.0, 0.0)
                    )
                ),
                spawn_chainage_m=None if raw.get("spawn_chainage_m") is None else float(raw["spawn_chainage_m"]),
                goal_chainage_m=None if raw.get("goal_chainage_m") is None else float(raw["goal_chainage_m"]),
                spawn_jitter_m=float(raw.get("spawn_jitter_m", 0.0)),
                collision_status=asset.collision_status,
            )
        )
    first = variants[0]
    return CaveWorldCfg(
        visual_asset_path=first.visual_asset_path,
        collision_asset_path=first.collision_asset_path,
        asset_scale=first.asset_scale,
        asset_translation_m=first.asset_translation_m,
        asset_orientation_wxyz=first.asset_orientation_wxyz,
        asset_name="Cave",
        clone_per_env=True,
        centerline_path=first.navigation_path,
        spawn_chainage_m=float(first.spawn_chainage_m or 0.0),
        goal_chainage_m=float(first.goal_chainage_m or 0.0),
        spawn_jitter_m=first.spawn_jitter_m,
        navigation_workspace_size_m=dataset.navigation_workspace_size_m,
        visual_enabled=bool(data.get("visual_enabled", True)),
        dataset_name=dataset.name,
        dataset_profile=profile,
        scene_variants=tuple(variants),
    )
