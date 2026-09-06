#!/usr/bin/env python3
"""Standalone checks for constant and time-varying current profiles."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.physics import CurrentField, CurrentMode, CurrentProfileCfg


def main() -> None:
    none = CurrentField(CurrentProfileCfg(mode=CurrentMode.NONE), 2, "cpu")
    assert torch.allclose(none.velocity(0.0), torch.zeros(2, 3))
    constant = CurrentField(
        CurrentProfileCfg(mode=CurrentMode.CONSTANT, mean_velocity_w_mps=(0.2, 0.0, 0.0)), 4, "cpu"
    )
    assert torch.allclose(constant.velocity(0.0), torch.tensor([[0.2, 0.0, 0.0]]).repeat(4, 1))
    sinusoidal = CurrentField(
        CurrentProfileCfg(
            mode=CurrentMode.SINUSOIDAL,
            mean_velocity_w_mps=(0.0, 0.0, 0.0),
            amplitude_w_mps=(0.4, 0.0, 0.0),
            period_s=10.0,
        ),
        2,
        "cpu",
    )
    assert not torch.allclose(sinusoidal.velocity(0.0), sinusoidal.velocity(2.5))
    random_walk = CurrentField(
        CurrentProfileCfg(mode=CurrentMode.RANDOM_WALK, random_walk_std_mps=0.1), 8, "cpu"
    )
    before = random_walk.velocity(0.0).clone()
    random_walk.step(0.1)
    assert not torch.allclose(before, random_walk.velocity(0.0))
    assert torch.linalg.vector_norm(random_walk.velocity(0.0), dim=-1).max() <= 0.5 + 1.0e-6
    random_constant = CurrentField(
        CurrentProfileCfg(mode=CurrentMode.RANDOM_CONSTANT, mean_velocity_w_mps=(0.1, 0.0, 0.0), random_walk_std_mps=0.1),
        4,
        "cpu",
    )
    before = random_constant.velocity(0.0).clone()
    random_constant.step(1.0)
    assert torch.allclose(before, random_constant.velocity(0.0))
    slow = CurrentField(
        CurrentProfileCfg(mode=CurrentMode.SLOW_VARYING, amplitude_w_mps=(0.4, 0.0, 0.0), period_s=10.0), 2, "cpu"
    )
    assert not torch.allclose(slow.velocity(0.0), slow.velocity(2.5))
    print("current_profiles: PASS")


if __name__ == "__main__":
    main()
