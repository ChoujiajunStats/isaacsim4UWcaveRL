#!/usr/bin/env python3
"""Analytic buoyancy checks that do not require Isaac Sim startup."""

from __future__ import annotations

import sys
from pathlib import Path

import math

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.physics.hydrodynamics import Hydrodynamics, HydrodynamicsCfg
from isaac_underwater.config import load_config


def vertical_acceleration(mass: float, volume: float, density: float, gravity: float) -> float:
    return (density * volume * gravity - mass * gravity) / mass


def main() -> None:
    robot_cfg = load_config("robot.yaml")
    assert robot_cfg["inertia_source"] == "uniform_box_approximation"
    assert len(robot_cfg["inertia_kg_m2"]) == 3
    assert all(value > 0.0 for value in robot_cfg["inertia_kg_m2"])
    mass = 20.0
    density = 1025.0
    gravity = 9.81
    neutral_volume = mass / density
    cases = {
        "negative": neutral_volume * 0.99,
        "neutral": neutral_volume,
        "positive": neutral_volume * 1.01,
    }
    acceleration = {name: vertical_acceleration(mass, volume, density, gravity) for name, volume in cases.items()}
    assert acceleration["negative"] < 0.0
    assert abs(acceleration["neutral"]) < 1.0e-6
    assert acceleration["positive"] > 0.0

    cfg = HydrodynamicsCfg(
        water_density_kg_m3=density,
        displaced_volume_m3=cases["positive"],
        gravity_mps2=gravity,
        center_of_buoyancy_m=(0.0, 0.0, 0.04),
    )
    assert torch.isclose(torch.tensor(cfg.buoyancy_n), torch.tensor(mass * gravity * 1.01), rtol=1.0e-5)

    roll = math.radians(20.0)
    quat = torch.tensor([[math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0]])
    zeros = torch.zeros(1, 3)
    _, restoring_torque = Hydrodynamics(cfg, "cpu").wrench_body(quat, zeros, zeros)
    assert restoring_torque[0, 0] * roll < 0.0
    for name, value in acceleration.items():
        print(f"{name}: vertical_acceleration={value:+.6f} m/s^2")
    print(f"restoring_roll_torque={restoring_torque[0, 0].item():+.6f} Nm")
    print("buoyancy: PASS")


if __name__ == "__main__":
    main()
