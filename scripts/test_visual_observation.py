#!/usr/bin/env python3
"""CPU contract test for the fixed-size stereo visual PPO observation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.navigation import build_visual_observation, visual_feature_dim


def main() -> None:
    batch = 3
    rgb_left = torch.randint(0, 256, (batch, 24, 32, 3), dtype=torch.uint8)
    rgb_right = torch.randint(0, 256, (batch, 24, 32, 3), dtype=torch.uint8)
    depth_left = torch.rand(batch, 24, 32, 1) * 20.0
    depth_right = torch.rand(batch, 24, 32, 1) * 20.0
    depth_left[0, 0, 0] = float("inf")
    observation = build_visual_observation(
        rgb_left,
        rgb_right,
        depth_left,
        depth_right,
        imu_acceleration=torch.zeros(batch, 3),
        imu_angular_velocity=torch.zeros(batch, 3),
        pressure_depth_m=torch.ones(batch, 1),
        previous_action=torch.zeros(batch, 6),
        mission_command=torch.zeros(batch, 4),
    )
    assert observation.shape == (batch, visual_feature_dim())
    assert torch.isfinite(observation).all()
    assert observation[:, : 8 * 12 * 16].abs().max() <= 1.0
    print(f"visual_feature_shape={tuple(observation.shape)}")
    print("visual_observation: PASS")


if __name__ == "__main__":
    main()
