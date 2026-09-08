#!/usr/bin/env python3
"""Pure PyTorch checks for label-free exploration state and depth safety."""

from __future__ import annotations

import torch

from isaac_underwater.navigation import VoxelVisitTracker, build_visual_observation, robust_forward_clearance


def main() -> None:
    tracker = VoxelVisitTracker(2, (2.0, 2.0, 2.0), 1.0, "cpu")
    start = torch.tensor([[-0.75, -0.75, -0.75], [0.75, 0.75, 0.75]])
    tracker.reset(torch.tensor([0, 1]), start)
    assert tracker.visited_count.tolist() == [1, 1]
    assert not tracker.update(start).any()

    moved = start.clone()
    moved[0, 0] = 0.25
    new_voxel = tracker.update(moved)
    assert new_voxel.tolist() == [True, False]
    assert tracker.visited_count.tolist() == [2, 1]
    assert torch.allclose(tracker.path_length_m, torch.tensor([1.0, 0.0]))

    tracker.reset(torch.tensor([0]), torch.tensor([[-0.75, -0.75, -0.75]]))
    assert tracker.visited_count.tolist() == [1, 1]
    assert tracker.workspace_coverage_fraction.tolist() == [0.125, 0.125]

    depth = torch.full((2, 8, 8, 1), 4.0)
    depth[0, 3:5, 3:5, 0] = 0.5
    depth[1, 3:5, 3:5, 0] = float("nan")
    clearance = robust_forward_clearance(depth, max_depth_m=6.0, crop_fraction=0.5, quantile=0.1)
    assert 0.49 <= float(clearance[0]) <= 0.51
    assert float(clearance[1]) == 4.0

    batch = 2
    rgb = torch.zeros(batch, 24, 32, 4, dtype=torch.uint8)
    visual_depth = torch.full((batch, 24, 32, 1), 4.0)
    visual = build_visual_observation(
        rgb,
        rgb,
        visual_depth,
        visual_depth,
        imu_acceleration=torch.zeros(batch, 3),
        imu_angular_velocity=torch.zeros(batch, 3),
        pressure_depth_m=torch.zeros(batch, 1),
        previous_action=torch.zeros(batch, 6),
        mission_command=None,
        output_hw=(12, 16),
    )
    assert visual.shape == (batch, 1549)
    print("exploration_contract: PASS")


if __name__ == "__main__":
    main()
