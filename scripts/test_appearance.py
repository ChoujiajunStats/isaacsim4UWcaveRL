#!/usr/bin/env python3
"""Standalone test for the renderer-independent underwater appearance gap."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.appearance import UnderwaterAppearanceCfg, UnderwaterLightingCfg, apply_underwater_appearance, lighting_intensity


def main() -> None:
    rgb = torch.ones(2, 3, 3, 3, dtype=torch.uint8) * 255
    depth = torch.full((2, 3, 3, 1), 4.0)
    cfg = UnderwaterAppearanceCfg(
        enabled=True,
        attenuation_rgb_per_m=(0.2, 0.1, 0.05),
        backscatter_strength=0.25,
        contrast=1.0,
    )
    transformed = apply_underwater_appearance(rgb, depth, cfg)
    assert transformed.dtype == torch.uint8
    assert transformed.shape == rgb.shape
    assert transformed[..., 0].float().mean() < transformed[..., 2].float().mean()
    batch_transformed = apply_underwater_appearance(
        rgb,
        depth,
        cfg,
        visibility_range_m=torch.tensor([4.0, 8.0]),
        attenuation_scale=torch.tensor([1.0, 1.5]),
        backscatter_strength=torch.tensor([0.1, 0.3]),
        exposure_offset=torch.tensor([-0.5, 0.5]),
        motion_blur_strength=torch.tensor([0.0, 0.4]),
    )
    assert batch_transformed.shape == rgb.shape
    assert not torch.equal(batch_transformed[1], transformed[1])
    assert torch.all(apply_underwater_appearance(rgb, depth, UnderwaterAppearanceCfg()) == rgb)
    assert lighting_intensity(100.0, UnderwaterLightingCfg(flicker_amplitude=0.2), 0.25) > 100.0
    print("appearance_gap: PASS")


if __name__ == "__main__":
    main()
