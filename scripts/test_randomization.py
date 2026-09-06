#!/usr/bin/env python3
"""Standalone sim-to-real randomization range checks."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.randomization import DomainGapCfg, DynamicsGapCfg, SensorGapCfg, sample_domain_gap, validate_physical_samples


def main() -> None:
    cfg = DomainGapCfg(
        dynamics=DynamicsGapCfg(mass_scale=(0.9, 1.1), current_speed_mps=(0.0, 0.4)),
        sensor=SensorGapCfg(imu_noise_std=(0.001, 0.01)),
    )
    samples = sample_domain_gap(cfg, 64)
    assert {"center_offset_m", "command_noise_std", "depth_noise_std", "relocalization_delay_s"} <= samples.keys()
    assert samples["mass_scale"].shape == (64,)
    assert samples["mass_scale"].min() >= 0.9
    assert samples["mass_scale"].max() <= 1.1
    assert samples["imu_noise_std"].min() >= 0.001
    assert samples["imu_noise_std"].max() <= 0.01
    assert bool(validate_physical_samples(samples).all())
    print(f"randomized_keys={len(samples)} environments=64")
    print("randomization: PASS")


if __name__ == "__main__":
    main()
