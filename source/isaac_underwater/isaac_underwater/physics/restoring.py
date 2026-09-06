"""Hydrostatic restoring helper separated from integration."""

from __future__ import annotations

import torch

from .buoyancy import buoyancy_wrench_body


def restoring_wrench_body(
    quat_wxyz: torch.Tensor,
    center_of_buoyancy_b: torch.Tensor,
    buoyancy_n: float,
    *,
    scale: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the external buoyancy wrench used alongside PhysX gravity."""
    return buoyancy_wrench_body(quat_wxyz, center_of_buoyancy_b, buoyancy_n, scale=scale)
