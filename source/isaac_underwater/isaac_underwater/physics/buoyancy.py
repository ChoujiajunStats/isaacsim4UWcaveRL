"""Buoyancy and hydrostatic restoring terms."""

from __future__ import annotations

import torch

from .hydro_utils import as_batch_scale


def buoyancy_wrench_body(
    quat_wxyz: torch.Tensor,
    center_of_buoyancy_b: torch.Tensor,
    buoyancy_n: float,
    *,
    scale: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upward buoyancy force and its CoB moment in body coordinates.

    Gravity is deliberately absent here: the PhysX rigid body owns gravity and
    therefore owns the weight term.  Applying only buoyancy avoids double
    counting the rigid-body force in the Isaac integration path.
    """
    batch_shape = quat_wxyz.shape[:-1]
    dtype = quat_wxyz.dtype
    device = quat_wxyz.device
    scale_t = as_batch_scale(scale, batch_shape, dtype=dtype, device=device)
    force_w = torch.zeros(*batch_shape, 3, dtype=dtype, device=device)
    force_w[..., 2] = buoyancy_n * scale_t
    from .hydro_utils import rotate_world_to_body

    force_b = rotate_world_to_body(quat_wxyz, force_w)
    torque_b = torch.linalg.cross(center_of_buoyancy_b, force_b, dim=-1)
    return force_b, torque_b
