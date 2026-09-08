#!/usr/bin/env python3
"""Convert an external cave visual/collision pair into Isaac USD.

The source meshes stay in ``ISAAC_UNDERWATER_ASSET_ROOT``.  Generated USDs
are written below ``outputs/cave_assets`` (ignored by git) and a manifest
records the exact source paths and conversion status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="worlds/porth_yr_ogof_sump9.yaml")
parser.add_argument("--force", action="store_true")
parser.add_argument("--usd-root", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg

from isaac_underwater.worlds import resolve_cave_asset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, destination: Path, *, collision: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    mesh_collision = schemas_cfg.TriangleMeshPropertiesCfg() if collision else None
    converter = MeshConverter(
        MeshConverterCfg(
            asset_path=str(source),
            usd_dir=str(destination.parent),
            usd_file_name=destination.name,
            force_usd_conversion=args.force,
            # Isaac Lab 2.3.2 emits a broken relative Props/ reference for
            # standalone mesh conversion in this layout.  Environment cloning
            # still shares the source USD layer, so keep the mesh self-contained
            # until that converter issue is resolved upstream.
            make_instanceable=False,
            collision_props=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True) if collision else None,
            mesh_collision_props=mesh_collision,
            mass_props=None,
            rigid_props=None,
        )
    )
    print(f"converted {source} -> {converter.usd_path}", flush=True)
    return Path(converter.usd_path)


def main() -> None:
    asset = resolve_cave_asset(args.config)
    output_root = Path(
        args.usd_root
        or PROJECT_ROOT / "outputs" / "cave_assets"
    ).expanduser().resolve()
    output_dir = output_root / asset.name
    visual_usd = convert(asset.visual_source, output_dir / "visual.usd", collision=False)
    collision_usd = convert(asset.collision_source, output_dir / "collision.usd", collision=True)
    manifest = {
        "name": asset.name,
        "visual_source": str(asset.visual_source),
        "collision_source": str(asset.collision_source),
        "visual_sha256": sha256(asset.visual_source),
        "collision_sha256": sha256(asset.collision_source),
        "visual_usd": str(visual_usd),
        "collision_usd": str(collision_usd),
        "metadata": str(asset.metadata_path) if asset.metadata_path else None,
        "centerline": str(asset.centerline_path) if asset.centerline_path else None,
        "scale_status": asset.scale_status,
        "candidate_status": asset.candidate_status,
        "collision_model": "provided_mesh_triangle_collision",
        "validation_status": "NOT_YET_VALIDATED_IN_ISAAC",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
