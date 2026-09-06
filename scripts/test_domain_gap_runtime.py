#!/usr/bin/env python3
"""Check YAML domain-gap parsing and per-environment application hooks."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.appearance import UnderwaterLightingCfg, lighting_intensity
from isaac_underwater.config import load_config
from isaac_underwater.physics import Hydrodynamics, HydrodynamicsCfg
from isaac_underwater.randomization import (
    domain_gap_from_mapping,
    perturb_thruster_command,
    rotate_body_vectors_z,
    sample_domain_gap,
)


def main() -> None:
    cfg = domain_gap_from_mapping(load_config("rl/domain_gap.yaml"))
    samples = sample_domain_gap(cfg, 4)
    assert "actuator_latency_s" in samples
    assert len(samples) == 35
    assert "added_mass_scale" in samples

    command = torch.ones(4, 8)
    samples["thruster_strength_scale"].fill_(1.0)
    samples["thruster_asymmetry_scale"].fill_(1.0)
    samples["command_noise_std"].zero_()
    samples["actuator_saturation_scale"].fill_(0.25)
    assert torch.allclose(perturb_thruster_command(command, samples), torch.full((4, 8), 0.25))

    hydro = Hydrodynamics(HydrodynamicsCfg(added_mass_kg=(2.0, 0.0, 0.0)), "cpu")
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    velocity = torch.zeros(2, 3)
    acceleration = torch.ones(2, 3)
    force, _ = hydro.wrench_body(
        quat,
        velocity,
        velocity,
        linear_acceleration_b=acceleration,
        buoyancy_scale=torch.tensor([1.0, 0.5]),
        drag_scale=torch.tensor([1.0, 2.0]),
        center_of_buoyancy_offset_m=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.02]]),
        added_mass_scale=torch.tensor([1.0, 2.0]),
    )
    assert force.shape == (2, 3)
    assert force[1, 2] < force[0, 2]

    rotated = rotate_body_vectors_z(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([90.0]))
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0]]), atol=1.0e-6)

    lighting_cfg = UnderwaterLightingCfg(distance_attenuation=0.2)
    assert lighting_intensity(100.0, lighting_cfg, distance_m=5.0) < 100.0
    print(f"domain_gap_keys={len(samples)} environments=4")
    print("domain_gap_runtime: PASS")


if __name__ == "__main__":
    main()
