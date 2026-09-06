#!/usr/bin/env python3
"""Standalone BlueROV thruster direction, allocation, saturation, and lag checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.actuators import ThrusterAllocator, ThrusterCfg, ThrusterCurve, ThrusterCurveCfg, ThrusterModel, bluerov2_thruster_layout


def main() -> None:
    model = ThrusterModel(bluerov2_thruster_layout(), "cpu")
    allocator = ThrusterAllocator(model)
    assert model.count == 8
    assert torch.linalg.matrix_rank(allocator.allocation_matrix).item() == 6

    unit_thrust = torch.eye(model.count)
    for index, cfg in enumerate(model.cfg):
        force, torque = model.wrench_from_thrust(unit_thrust[index : index + 1])
        expected_force = model.directions[index]
        expected_torque = torch.linalg.cross(model.positions[index], expected_force, dim=-1)
        assert torch.allclose(force[0], expected_force, atol=1.0e-6)
        assert torch.allclose(torque[0], expected_torque, atol=1.0e-6)
        print(f"{cfg.name}: force={force[0].tolist()} torque={torque[0].tolist()} PASS")

    desired_force = torch.tensor([[10.0, -6.0, 8.0]])
    desired_torque = torch.tensor([[1.5, -1.0, 2.0]])
    allocated = allocator.allocate_thrust(desired_force, desired_torque)
    actual_force, actual_torque = model.wrench_from_thrust(allocated)
    assert torch.allclose(actual_force, desired_force, atol=1.0e-4)
    assert torch.allclose(actual_torque, desired_torque, atol=1.0e-4)

    saturated = model.command_to_target_thrust(torch.full((1, model.count), 2.0))
    assert torch.allclose(saturated, model.forward.unsqueeze(0))
    reverse = model.command_to_target_thrust(torch.full((1, model.count), -2.0))
    assert torch.allclose(reverse, -model.reverse.unsqueeze(0))
    model.reset(1)
    first_step = model.step(torch.ones(1, model.count), dt=0.01)
    assert torch.all(first_step > 0.0)
    assert torch.all(first_step < model.forward)
    linear = ThrusterCurve(ThrusterCurveCfg(kind="linear"), "cpu")
    assert torch.allclose(linear.evaluate(torch.tensor([0.25])), torch.tensor([0.25]))
    measured = ThrusterCurve(
        ThrusterCurveCfg(
            kind="measured_curve",
            command_points=(0.0, 0.5, 1.0),
            forward_points=(0.0, 0.2, 1.0),
            reverse_points=(0.0, 0.1, 0.8),
        ),
        "cpu",
    )
    assert torch.allclose(measured.evaluate(torch.tensor([0.75])), torch.tensor([0.6]))
    assert torch.allclose(measured.evaluate(torch.tensor([0.75]), reverse=torch.tensor([True])), torch.tensor([0.45]))
    print("curve_types: PASS")
    print("allocation/saturation/response_lag: PASS")
    print("thrusters: PASS")


if __name__ == "__main__":
    main()
