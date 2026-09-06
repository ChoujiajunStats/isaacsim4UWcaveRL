"""Fossen-style relative-velocity damping."""

from __future__ import annotations

import torch


def linear_quadratic_damping(
    nu_r: torch.Tensor,
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
) -> torch.Tensor:
    """Return ``-D_l nu_r - D_q |nu_r| nu_r`` for all six axes."""
    return -linear_coefficients * nu_r - quadratic_coefficients * nu_r.abs() * nu_r


def damping_matrix(
    linear_coefficients: torch.Tensor,
    quadratic_coefficients: torch.Tensor,
    nu_r: torch.Tensor,
) -> torch.Tensor:
    """Return the instantaneous positive diagonal damping matrix."""
    diagonal = linear_coefficients + quadratic_coefficients * nu_r.abs()
    return torch.diag_embed(diagonal)
