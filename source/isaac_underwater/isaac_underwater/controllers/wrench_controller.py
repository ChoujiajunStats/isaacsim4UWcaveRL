"""Normalized policy action to body wrench mapping."""

from __future__ import annotations

import torch


class WrenchController:
    """Map normalized commands to body wrench.

    Four-dimensional legacy actions retain ``Fx,Fy,Fz,Tz`` semantics.  A
    six-dimensional action uses the full ``Fx,Fy,Fz,Tx,Ty,Tz`` wrench and is
    passed directly to the existing six-axis allocator.
    """

    def __init__(
        self,
        max_force_n: tuple[float, float, float],
        max_yaw_torque_nm: float,
        device: str,
        max_torque_nm: tuple[float, float, float] | None = None,
    ):
        self.max_force = torch.tensor(max_force_n, dtype=torch.float32, device=device)
        self.max_torque = torch.tensor(
            max_torque_nm if max_torque_nm is not None else (max_yaw_torque_nm, max_yaw_torque_nm, max_yaw_torque_nm),
            dtype=torch.float32,
            device=device,
        )
        self.max_yaw_torque = float(max_yaw_torque_nm)

    def map_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clipped = action.clamp(-1.0, 1.0)
        force = clipped[:, :3] * self.max_force
        torque = torch.zeros_like(force)
        if clipped.shape[-1] == 6:
            torque = clipped[:, 3:6] * self.max_torque
        elif clipped.shape[-1] == 4:
            torque[:, 2] = clipped[:, 3] * self.max_yaw_torque
        else:
            raise ValueError(f"Expected 4 or 6 wrench action dimensions, got {clipped.shape[-1]}")
        return force, torque
