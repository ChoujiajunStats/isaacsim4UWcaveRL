#!/usr/bin/env python3
"""Check the shared velocity, wrench, and direct-thruster command paths."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.actuators import ThrusterAllocator, ThrusterModel, bluerov2_thruster_layout
from isaac_underwater.controllers import VelocityController, WrenchController, command_to_thruster
from isaac_underwater.interfaces import ControlCommand


def main() -> None:
    model = ThrusterModel(bluerov2_thruster_layout(), "cpu")
    allocator = ThrusterAllocator(model)
    controller = VelocityController((35.0, 40.0, 45.0), 12.0, (80.0, 80.0, 100.0), 25.0, "cpu")
    commands = (
        ControlCommand.from_body_velocity(0.0, torch.ones(1, 3), torch.zeros(1)),
        ControlCommand.from_body_wrench(0.0, torch.zeros(1, 6)),
        ControlCommand.from_thrusters(0.0, torch.full((1, 8), 0.2)),
    )
    for command in commands:
        output = command_to_thruster(
            command,
            velocity_controller=controller,
            allocator=allocator,
            linear_velocity_b=torch.zeros(1, 3),
            yaw_rate=torch.zeros(1),
        )
        assert output.shape == (1, 8)
        assert torch.all(output.abs() <= 1.0)
    wrench_controller = WrenchController(
        (80.0, 80.0, 100.0),
        25.0,
        "cpu",
        max_torque_nm=(10.0, 12.0, 25.0),
    )
    force, torque = wrench_controller.map_action(torch.tensor([[0.2, -0.3, 0.4, 0.5, -0.6, 0.7]]))
    six_dof = command_to_thruster(
        ControlCommand.from_body_wrench(0.0, torch.cat((force, torque), dim=-1)),
        velocity_controller=controller,
        allocator=allocator,
        linear_velocity_b=torch.zeros(1, 3),
        yaw_rate=torch.zeros(1),
    )
    assert six_dof.shape == (1, 8) and torch.all(six_dof.abs() <= 1.0)
    print("command_modes: PASS")


if __name__ == "__main__":
    main()
