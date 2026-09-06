"""Added-mass matrix and energy-neutral Coriolis approximation."""

from __future__ import annotations

import torch

from .hydro_utils import skew


def diagonal_added_mass(translational: torch.Tensor, rotational: torch.Tensor, device, dtype) -> torch.Tensor:
    values = torch.cat((translational, rotational)).to(device=device, dtype=dtype)
    return torch.diag(values)


def added_mass_coriolis(matrix: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Build a skew-symmetric ``C_A`` approximation from block momentum.

    The block form is valid for diagonal and block-diagonal added mass.  For a
    full identified matrix it remains an energy-neutral approximation and is
    therefore exposed only by the ``reference`` preset.
    """
    a = torch.matmul(matrix[..., :3, :], nu.unsqueeze(-1)).squeeze(-1)
    b = torch.matmul(matrix[..., 3:, :], nu.unsqueeze(-1)).squeeze(-1)
    zeros = torch.zeros_like(skew(a))
    return torch.cat(
        (
            torch.cat((zeros, -skew(a)), dim=-1),
            torch.cat((-skew(a), -skew(b)), dim=-1),
        ),
        dim=-2,
    )
