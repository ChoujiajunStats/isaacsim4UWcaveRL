"""Cheap, deterministic underwater image-gap transforms for rendered sensors."""

from __future__ import annotations

import torch

from .lighting import UnderwaterAppearanceCfg


def apply_underwater_appearance(
    rgb: torch.Tensor,
    depth: torch.Tensor | None,
    cfg: UnderwaterAppearanceCfg,
    *,
    visibility_range_m: torch.Tensor | float | None = None,
    attenuation_scale: torch.Tensor | float | None = None,
    backscatter_strength: torch.Tensor | float | None = None,
    exposure_offset: torch.Tensor | float | None = None,
    motion_blur_strength: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Apply attenuation, color absorption, backscatter, exposure, and contrast.

    The transform is intentionally post-render and vectorized. It keeps the
    physics-only mode free of renderer work while making the appearance gap
    explicit and reproducible for perception experiments.
    """
    if not cfg.enabled or depth is None:
        return rgb
    original_dtype = rgb.dtype
    image = rgb.float()
    scale = 255.0 if original_dtype == torch.uint8 else 1.0
    image = image / scale
    if image.shape[-1] < 3:
        raise ValueError("RGB image must have at least three channels")
    distance = depth.float()
    if distance.shape[-1] == 1:
        distance = distance[..., 0]
    distance = distance.unsqueeze(-1).clamp_min(0.0)

    def override(value: torch.Tensor | float | None, default: float) -> torch.Tensor:
        result = torch.as_tensor(default if value is None else value, device=rgb.device, dtype=image.dtype)
        if result.ndim == 0:
            return result
        return result.reshape(result.shape[0], *([1] * (distance.ndim - 1)))

    attenuation = torch.as_tensor(cfg.attenuation_rgb_per_m, device=rgb.device, dtype=image.dtype)
    attenuation = attenuation * override(attenuation_scale, 1.0)
    fog_transmission = torch.exp(-cfg.fog_density * distance)
    visibility = override(visibility_range_m, cfg.visibility_range_m).clamp_min(1.0e-6)
    range_transmission = torch.exp(-distance / visibility)
    transmittance = torch.exp(-distance * attenuation) * fog_transmission * range_transmission
    water_color = torch.as_tensor(cfg.water_color_rgb, device=rgb.device, dtype=image.dtype)
    backscatter = (
        override(backscatter_strength, cfg.backscatter_strength)
        * (1.0 - transmittance)
        * water_color
        * cfg.background_intensity
    )
    image_rgb = image[..., :3] * transmittance + backscatter
    exposure = cfg.camera_exposure + override(exposure_offset, 0.0)
    image_rgb = image_rgb * (2.0**exposure)
    image_rgb = (image_rgb - 0.5) * cfg.contrast + 0.5
    # A cheap image-space approximation keeps motion-blur randomization
    # available without adding renderer state or extra camera buffers.
    blur = override(motion_blur_strength, 0.0).clamp(0.0, 1.0)
    if torch.any(blur > 0.0):
        neighboring = 0.5 * (
            torch.roll(image_rgb, shifts=1, dims=-2)
            + torch.roll(image_rgb, shifts=1, dims=-3)
        )
        image_rgb = image_rgb * (1.0 - blur) + neighboring * blur
    if image.shape[-1] > 3:
        image = torch.cat((image_rgb, image[..., 3:]), dim=-1)
    else:
        image = image_rgb
    image = image.clamp(0.0, 1.0)
    return (image * scale).round().to(original_dtype) if original_dtype == torch.uint8 else image
