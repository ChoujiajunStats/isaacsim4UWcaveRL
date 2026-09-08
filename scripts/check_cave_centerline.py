#!/usr/bin/env python3
"""Check centerline-to-collision-mesh clearance for cave spawn candidates.

This is a geometric pre-check, not a proof of collision-free navigation.  The
photogrammetry mesh may be open and the robot is approximated by a bounding
sphere, so the report deliberately keeps those limitations explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import trimesh
import yaml

from isaac_underwater.worlds import resolve_cave_asset


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="worlds/porth_yr_ogof_sump9.yaml")
parser.add_argument("--robot-radius", type=float, default=0.5)
parser.add_argument("--output", default=None)
args = parser.parse_args()


def main() -> None:
    asset = resolve_cave_asset(args.config)
    if asset.centerline_path is None:
        raise ValueError(f"No centerline configured for {asset.name}")
    mesh = trimesh.load(asset.collision_source, force="mesh", process=False)
    rows = list(csv.DictReader(asset.centerline_path.open(encoding="utf-8")))
    points = np.asarray(
        [[float(row[key]) for key in ("north_m", "east_m", "down_m")] for row in rows], dtype=np.float64
    )
    chainage = np.asarray([float(row["chainage_m"]) for row in rows], dtype=np.float64)
    clearance = np.asarray(mesh.nearest.on_surface(points)[1], dtype=np.float64)

    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    spawn_chainage = float(config.get("spawn_chainage_m", chainage[0]))
    goal_chainage = float(config.get("goal_chainage_m", chainage[-1]))
    spawn_index = int(np.argmin(np.abs(chainage - spawn_chainage)))
    goal_index = int(np.argmin(np.abs(chainage - goal_chainage)))
    output = {
        "asset": asset.name,
        "collision_mesh": str(asset.collision_source),
        "watertight": bool(mesh.is_watertight),
        "robot_bounding_sphere_radius_m": args.robot_radius,
        "centerline_points": int(len(points)),
        "centerline_min_clearance_m": float(np.min(clearance)),
        "centerline_p01_clearance_m": float(np.percentile(clearance, 1)),
        "spawn": {
            "requested_chainage_m": spawn_chainage,
            "actual_chainage_m": float(chainage[spawn_index]),
            "position_m": points[spawn_index].tolist(),
            "clearance_m": float(clearance[spawn_index]),
            "status": "PASS" if clearance[spawn_index] >= args.robot_radius else "FAIL",
        },
        "goal": {
            "requested_chainage_m": goal_chainage,
            "actual_chainage_m": float(chainage[goal_index]),
            "position_m": points[goal_index].tolist(),
            "clearance_m": float(clearance[goal_index]),
            "status": "PASS" if clearance[goal_index] >= args.robot_radius else "FAIL",
        },
        "full_centerline_clearance_status": "PASS"
        if bool(np.all(clearance >= args.robot_radius))
        else "NOT_YET_VALIDATED",
        "limitations": [
            "centerline is an external provisional route annotation",
            "open/non-watertight mesh does not prove free-space connectivity",
            "bounding-sphere clearance is conservative but not a swept-volume check",
        ],
    }
    output_path = Path(args.output or Path("logs") / "cave_clearance" / f"{asset.name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"cave_centerline_check: spawn={output['spawn']['status']} goal={output['goal']['status']}")


if __name__ == "__main__":
    main()
