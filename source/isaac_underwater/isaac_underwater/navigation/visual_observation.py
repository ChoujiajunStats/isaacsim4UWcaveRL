"""GPU-friendly low-resolution visual observation construction.

The first visual PPO pilot intentionally uses a fixed-size tensor rather than
passing raw 160x120 images through the existing MLP.  This is a transport and
training contract, not a claim that the handcrafted downsampling is the final
camera encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def visual_feature_dim(
    *,
    output_hw: tuple[int, int] = (12, 16),
    include_imu: bool = True,
    include_pressure: bool = True,
    include_previous_action: bool = True,
    action_dim: int = 6,
    mission_command_dim: int = 4,
) -> int:
    channels = 8  # left/right RGB (6) + left/right depth (2)
    scalar_dim = 0
    scalar_dim += 6 if include_imu else 0
    scalar_dim += 1 if include_pressure else 0
    scalar_dim += action_dim if include_previous_action else 0
    scalar_dim += mission_command_dim
    return channels * output_hw[0] * output_hw[1] + scalar_dim


def _rgb_to_nchw(image: torch.Tensor) -> torch.Tensor:
    value = image.float()
    if value.ndim != 4 or value.shape[-1] < 3:
        raise ValueError(f"Expected batched RGB image [N,H,W,C], got {tuple(image.shape)}")
    value = value[..., :3]
    if value.detach().amax() > 1.5:
        value = value / 255.0
    return value.clamp(0.0, 1.0).permute(0, 3, 1, 2)


def _depth_to_nchw(depth: torch.Tensor, max_depth_m: float) -> torch.Tensor:
    value = depth.float()
    if value.ndim != 4:
        raise ValueError(f"Expected batched depth image [N,H,W,C], got {tuple(depth.shape)}")
    if value.shape[-1] > 1:
        value = value[..., :1]
    finite = torch.isfinite(value)
    value = torch.nan_to_num(value, nan=max_depth_m, posinf=max_depth_m, neginf=0.0)
    value = value.clamp(0.0, max_depth_m) / max(max_depth_m, 1.0e-6)
    # Keep invalid pixels deterministic while preserving valid depth values.
    value = torch.where(finite, value, torch.ones_like(value))
    return value.permute(0, 3, 1, 2)


def build_visual_observation(
    rgb_left: torch.Tensor,
    rgb_right: torch.Tensor,
    depth_left: torch.Tensor,
    depth_right: torch.Tensor,
    *,
    imu_acceleration: torch.Tensor | None = None,
    imu_angular_velocity: torch.Tensor | None = None,
    pressure_depth_m: torch.Tensor | None = None,
    previous_action: torch.Tensor | None = None,
    mission_command: torch.Tensor | None = None,
    output_hw: tuple[int, int] = (12, 16),
    max_depth_m: float = 12.0,
) -> torch.Tensor:
    """Return a fixed-size normalized stereo visual observation on the input device."""
    images = torch.cat(
        (
            _rgb_to_nchw(rgb_left),
            _rgb_to_nchw(rgb_right),
            _depth_to_nchw(depth_left, max_depth_m),
            _depth_to_nchw(depth_right, max_depth_m),
        ),
        dim=1,
    )
    images = F.interpolate(images, size=output_hw, mode="bilinear", align_corners=False)
    features = [images.flatten(start_dim=1)]
    if imu_acceleration is not None:
        features.append(imu_acceleration.float())
    if imu_angular_velocity is not None:
        features.append(imu_angular_velocity.float())
    if pressure_depth_m is not None:
        features.append(pressure_depth_m.float())
    if previous_action is not None:
        features.append(previous_action.float())
    if mission_command is not None:
        features.append(mission_command.float())
    return torch.cat(features, dim=-1)
