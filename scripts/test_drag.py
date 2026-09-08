#!/usr/bin/env python3
"""Analytic quadratic-drag direction and dissipation checks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.physics.hydrodynamics import linear_quadratic_drag, quadratic_drag


def main() -> None:
    velocity = torch.tensor([[2.0, -1.5, 0.0], [-0.25, 0.5, 3.0]])
    coefficients = torch.tensor([12.0, 20.0, 25.0])
    force = quadratic_drag(velocity, coefficients)
    power = torch.sum(force * velocity, dim=-1)
    assert torch.all(force[velocity > 0.0] < 0.0)
    assert torch.all(force[velocity < 0.0] > 0.0)
    assert torch.all(force[velocity == 0.0] == 0.0)
    assert torch.all(power <= 0.0)
    mixed_force = linear_quadratic_drag(velocity, torch.tensor([1.0, 2.0, 3.0]), coefficients)
    mixed_power = torch.sum(mixed_force * velocity, dim=-1)
    assert torch.all(mixed_power <= power)
    print(f"drag_force={force.tolist()}")
    print(f"mechanical_power={power.tolist()}")
    print("drag: PASS")


if __name__ == "__main__":
    main()
