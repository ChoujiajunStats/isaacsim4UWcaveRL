#!/usr/bin/env python3
"""Fail when a multi-cave navigation evaluation misses acceptance targets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--min-episodes-per-scene", type=int, default=30)
    parser.add_argument("--min-success-rate", type=float, default=0.80)
    parser.add_argument("--max-collision-rate", type=float, default=0.15)
    parser.add_argument("--min-mean-spl", type=float, default=0.50)
    parser.add_argument("--expected-profile", default=None)
    parser.add_argument(
        "--dataset-config", type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/worlds/caves_difficulty_v01.yaml",
        help="Manifest defining the exact scene set required by each profile.",
    )
    parser.add_argument("--require-domain-randomization", action="store_true")
    return parser.parse_args()


def _rate(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1], got {result}")
    return result


def validate_metrics(
    metrics: object,
    *,
    expected_profile: str,
    expected_scenes: set[str],
    min_episodes: int = 30,
    min_success: float = 0.80,
    max_collision: float = 0.15,
    min_spl: float = 0.50,
    require_domain_randomization: bool = False,
) -> list[str]:
    """Fail closed on incomplete scenes and training-only curriculum results."""
    if not isinstance(metrics, dict):
        return ["Evaluation JSON must contain an object"]
    per_scene = metrics.get("per_scene")
    if not isinstance(per_scene, dict) or not per_scene:
        return ["Evaluation JSON has no non-empty per_scene metrics"]

    failures: list[str] = []
    actual_profile = metrics.get("cave_dataset_profile")
    if actual_profile != expected_profile:
        failures.append(f"profile: expected {expected_profile!r}, got {actual_profile!r}")
    missing = expected_scenes - set(per_scene)
    unexpected = set(per_scene) - expected_scenes
    if missing:
        failures.append(f"missing scenes: {sorted(missing)}")
    if unexpected:
        failures.append(f"unexpected scenes: {sorted(unexpected)}")
    if metrics.get("navigation_curriculum") is not False:
        failures.append("navigation_curriculum: must explicitly be false for entrance-to-exit acceptance")
    if require_domain_randomization and metrics.get("domain_randomization") is not True:
        failures.append("domain_randomization: expected true")

    scene_episodes = 0
    for scene_name, raw_values in per_scene.items():
        if not isinstance(raw_values, dict):
            failures.append(f"{scene_name}: metrics must be an object")
            continue
        episodes = raw_values.get("episodes")
        if type(episodes) is not int or episodes < min_episodes:
            failures.append(
                f"{scene_name}: episodes {episodes!r} must be an integer >= {min_episodes}"
            )
        if type(episodes) is int and episodes >= 0:
            scene_episodes += episodes
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

    if type(metrics.get("episodes")) is not int or metrics["episodes"] != scene_episodes:
        failures.append(f"episodes: total must match sum of per-scene counts ({scene_episodes})")
    return failures


def main() -> None:
    import yaml

    args = parse_args()
    if args.min_episodes_per_scene <= 0:
        raise ValueError("--min-episodes-per-scene must be positive")
    min_success = _rate(args.min_success_rate, "--min-success-rate")
    max_collision = _rate(args.max_collision_rate, "--max-collision-rate")
    min_spl = _rate(args.min_mean_spl, "--min-mean-spl")
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("Evaluation JSON must contain an object")
    manifest = yaml.safe_load(args.dataset_config.read_text(encoding="utf-8"))
    profile = args.expected_profile or metrics.get("cave_dataset_profile")
    if not isinstance(profile, str) or profile not in manifest["profiles"]:
        raise ValueError(f"Unknown dataset profile: {profile!r}")
    expected_scenes = manifest["profiles"][profile]
    if (
        not isinstance(expected_scenes, list) or not expected_scenes
        or any(not isinstance(scene, str) or scene not in manifest["scenes"] for scene in expected_scenes)
        or len(set(expected_scenes)) != len(expected_scenes)
    ):
        raise ValueError(f"Invalid scene list for profile: {profile!r}")
    failures = validate_metrics(
        metrics, expected_profile=profile, expected_scenes=set(expected_scenes),
        min_episodes=args.min_episodes_per_scene, min_success=min_success,
        max_collision=max_collision, min_spl=min_spl,
        require_domain_randomization=args.require_domain_randomization,
    )
    if failures:
        print("navigation_metrics: FAIL")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(
        "navigation_metrics: PASS "
        f"scenes={len(expected_scenes)} min_episodes={args.min_episodes_per_scene} "
        f"success>={min_success:.2f} collision<={max_collision:.2f} SPL>={min_spl:.2f}"
    )


if __name__ == "__main__":
    main()
