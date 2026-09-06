"""Weighted trajectory loss used by future reference/real-data fitting."""

from __future__ import annotations

import torch


def trajectory_loss(
    predicted: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
    velocity_weight: float = 1.0,
    angular_rate_weight: float = 1.0,
) -> torch.Tensor:
    """Compute a differentiable weighted MSE over available trajectory fields."""
    terms = []
    for key, weight in (
        ("position", position_weight),
        ("orientation", orientation_weight),
        ("velocity", velocity_weight),
        ("angular_velocity", angular_rate_weight),
    ):
        if key in predicted and key in target:
            terms.append(torch.as_tensor(weight, device=predicted[key].device) * (predicted[key] - target[key]).square().mean())
    if not terms:
        raise ValueError("No common trajectory fields available for loss")
    return torch.stack(terms).sum()
