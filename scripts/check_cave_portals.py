#!/usr/bin/env python3
"""Infer cave portals from geometry-derived skeleton clearance metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from isaac_underwater.navigation import infer_clearance_portals
from isaac_underwater.worlds import resolve_cave_asset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="worlds/porth_yr_ogof_sump9.yaml")
    parser.add_argument("--minimum-clearance-m", type=float, default=0.65)
    parser.add_argument("--minimum-exterior-run-m", type=float, default=1.0)
    parser.add_argument("--tangent-probe-m", type=float, default=1.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    asset = resolve_cave_asset(args.config, require_sources=False)
    if asset.centerline_path is None or not asset.centerline_path.is_file():
        raise FileNotFoundError(f"Geometry-derived centerline is missing for {asset.name}")
    rows = list(csv.DictReader(asset.centerline_path.open(encoding="utf-8")))
    required = ("chainage_m", "north_m", "east_m", "down_m", "nominal_surface_clearance_m")
    if not rows or any(field not in rows[0] for field in required):
        raise ValueError(f"Centerline must contain {required}: {asset.centerline_path}")
    points = torch.tensor(
        [[float(row[key]) for key in ("north_m", "east_m", "down_m")] for row in rows],
        dtype=torch.float32,
    )
    chainage = torch.tensor([float(row["chainage_m"]) for row in rows], dtype=torch.float32)
    clearance = torch.tensor(
        [float(row["nominal_surface_clearance_m"]) for row in rows], dtype=torch.float32
    )
    portals = infer_clearance_portals(
        points,
        chainage,
        clearance,
        minimum_clearance_m=args.minimum_clearance_m,
        minimum_exterior_run_m=args.minimum_exterior_run_m,
        tangent_probe_m=args.tangent_probe_m,
    )
    if not portals:
        raise RuntimeError("No portal satisfied the automatic inference contract")

    report = {
        "asset": asset.name,
        "status": "PASS_PROVISIONAL_GEOMETRY_INFERENCE",
        "method": "automatic_skeleton_endpoint_clearance_transition",
        "manual_entrance_coordinate_used": False,
        "source_centerline": str(asset.centerline_path),
        "source_candidate_status": asset.candidate_status,
        "requirements": {
            "minimum_clearance_m": args.minimum_clearance_m,
            "minimum_exterior_run_m": args.minimum_exterior_run_m,
            "tangent_probe_m": args.tangent_probe_m,
        },
        "portals": [
            {
                "endpoint": portal.endpoint,
                "transition_index": portal.transition_index,
                "chainage_m": portal.chainage_m,
                "exterior_run_m": portal.exterior_run_m,
                "local_clearance_m": portal.local_clearance_m,
                "position_m": portal.position_m.tolist(),
                "inward_direction": portal.inward_direction.tolist(),
            }
            for portal in portals
        ],
    }
    output = args.output or PROJECT_ROOT / "logs" / "cave_portals" / f"{asset.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"cave_portal_check: PASS asset={asset.name} candidates={len(portals)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
