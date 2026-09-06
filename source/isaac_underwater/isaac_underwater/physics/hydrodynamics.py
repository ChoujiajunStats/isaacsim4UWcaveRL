"""Backward-compatible wrapper around the Fossen marine dynamics layer."""

from __future__ import annotations

import torch

from .damping import linear_quadratic_damping
from .hydro_utils import rotate_body_to_world, rotate_world_to_body
from .marine_dynamics import MarineDynamics, MarineDynamicsCfg


HydrodynamicsCfg = MarineDynamicsCfg


def linear_quadratic_drag(
    velocity: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Compatibility alias for the six-axis damping primitive."""
    return linear_quadratic_damping(velocity, linear_coefficients, quadratic_coefficients)


def quadratic_drag(velocity: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    """Compatibility helper for quadratic-only drag."""
    return linear_quadratic_damping(velocity, torch.zeros_like(coefficients), coefficients)


class Hydrodynamics(MarineDynamics):
    """Legacy class name; no separate rigid-body integration is performed."""
