#!/usr/bin/env python3
"""Fail when a multi-cave navigation evaluation misses acceptance targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("metrics", type=Path)
parser.add_argument("--min-episodes-per-scene", type=int, default=30)
parser.add_argument("--min-success-rate", type=float, default=0.80)
parser.add_argument("--max-collision-rate", type=float, default=0.15)
parser.add_argument("--min-mean-spl", type=float, default=0.50)
parser.add_argument("--expected-profile", default=None)
parser.add_argument("--require-domain-randomization", action="store_true")
args = parser.parse_args()


def _rate(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1], got {result}")
    return result


def main() -> None:
    if args.min_episodes_per_scene <= 0:
        raise ValueError("--min-episodes-per-scene must be positive")
    min_success = _rate(args.min_success_rate, "--min-success-rate")
    max_collision = _rate(args.max_collision_rate, "--max-collision-rate")
    min_spl = _rate(args.min_mean_spl, "--min-mean-spl")
    if not args.metrics.is_file():
        raise FileNotFoundError(args.metrics)

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("Evaluation JSON must contain an object")
    per_scene = metrics.get("per_scene")
    if not isinstance(per_scene, dict) or not per_scene:
        raise ValueError("Evaluation JSON has no non-empty per_scene metrics")

    failures: list[str] = []
    if args.expected_profile is not None:
        actual_profile = metrics.get("cave_dataset_profile")
        if actual_profile != args.expected_profile:
            failures.append(
                f"profile: expected {args.expected_profile!r}, got {actual_profile!r}"
            )
    if args.require_domain_randomization and not bool(metrics.get("domain_randomization")):
        failures.append("domain_randomization: expected true")

    for scene_name, raw_values in per_scene.items():
        if not isinstance(raw_values, dict):
            failures.append(f"{scene_name}: metrics must be an object")
            continue
        episodes = raw_values.get("episodes")
        if not isinstance(episodes, int) or episodes < args.min_episodes_per_scene:
            failures.append(
                f"{scene_name}: episodes {episodes!r} < {args.min_episodes_per_scene}"
            )
        try:
            success = _rate(raw_values.get("success_rate"), f"{scene_name}.success_rate")
            collision = _rate(
                raw_values.get("collision_rate"), f"{scene_name}.collision_rate"
            )
            spl = _rate(raw_values.get("mean_spl"), f"{scene_name}.mean_spl")
        except ValueError as error:
            failures.append(str(error))
            continue
        if success < min_success:
            failures.append(f"{scene_name}: success_rate {success:.3f} < {min_success:.3f}")
        if collision > max_collision:
            failures.append(
                f"{scene_name}: collision_rate {collision:.3f} > {max_collision:.3f}"
            )
        if spl < min_spl:
            failures.append(f"{scene_name}: mean_spl {spl:.3f} < {min_spl:.3f}")

    if failures:
        print("navigation_metrics: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(
        "navigation_metrics: PASS "
        f"scenes={len(per_scene)} min_episodes={args.min_episodes_per_scene} "
        f"success>={min_success:.2f} collision<={max_collision:.2f} SPL>={min_spl:.2f}"
    )


if __name__ == "__main__":
    main()
