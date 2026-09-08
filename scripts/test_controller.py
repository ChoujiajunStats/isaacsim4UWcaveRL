#!/usr/bin/env python3
"""Standalone body-velocity controller and allocator checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.actuators import ThrusterAllocator, ThrusterModel, bluerov2_thruster_layout
from isaac_underwater.controllers import VelocityController


def main() -> None:
    controller = VelocityController(
        linear_gain=(35.0, 40.0, 45.0),
        yaw_gain=12.0,
        max_force_n=(80.0, 80.0, 100.0),
        max_yaw_torque_nm=25.0,
        device="cpu",
    )
    desired = torch.tensor([[1.0, -0.5, 0.25]])
    force, torque = controller.compute(desired, torch.tensor([0.5]), torch.zeros(1, 3), torch.zeros(1))
    assert torch.allclose(force, torch.tensor([[35.0, -20.0, 11.25]]))
    assert torch.allclose(torque, torch.tensor([[0.0, 0.0, 6.0]]))
    saturated, saturated_torque = controller.compute(
        torch.full((1, 3), 10.0), torch.tensor([10.0]), torch.zeros(1, 3), torch.zeros(1)
    )
    assert torch.all(saturated <= torch.tensor([[80.0, 80.0, 100.0]]))
    assert torch.allclose(saturated_torque[:, 2], torch.tensor([25.0]))

    model = ThrusterModel(bluerov2_thruster_layout(), "cpu")
    allocator = ThrusterAllocator(model)
    commands = allocator.allocate_command(force, torque)
    assert commands.shape == (1, 8)
    assert torch.all(commands.abs() <= 1.0)
    print("controller: PASS")


if __name__ == "__main__":
    main()
