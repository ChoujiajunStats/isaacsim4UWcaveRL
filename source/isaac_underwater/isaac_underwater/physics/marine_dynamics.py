"""Fossen-compatible, wrench-only marine craft dynamics.

This module computes the terms that are added to an Isaac rigid body.  It does
not integrate pose or velocity.  PhysX remains responsible for rigid-body
mass/inertia, gravity, integration, and contacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .added_mass import added_mass_coriolis, diagonal_added_mass
from .buoyancy import buoyancy_wrench_body
from .damping import damping_matrix, linear_quadratic_damping
from .hydro_utils import as_batch_scale, skew, validate_symmetric_positive_definite
from .hydro_utils import rotate_world_to_body


Vector3 = tuple[float, float, float]
Preset = Literal["fast_rl", "hydro_rl", "reference"]


@dataclass(frozen=True)
class MarineDynamicsCfg:
    water_density_kg_m3: float = 1025.0
    displaced_volume_m3: float = 0.0195
    gravity_mps2: float = 9.81
    mass_kg: float = 20.0
    inertia_kg_m2: Vector3 = (0.5666667, 1.2166667, 1.4833333)
    center_of_mass_m: Vector3 = (0.0, 0.0, 0.0)
    center_of_buoyancy_m: Vector3 = (0.0, 0.0, 0.03)
    linear_drag_coeff: Vector3 = (12.0, 20.0, 25.0)
    quadratic_drag_coeff: Vector3 = (18.0, 28.0, 35.0)
    angular_linear_drag_coeff: Vector3 = (1.5, 1.5, 2.0)
    angular_quadratic_drag_coeff: Vector3 = (2.0, 2.0, 3.0)
    added_mass_kg: Vector3 = (0.0, 0.0, 0.0)
    added_inertia_kg_m2: Vector3 = (0.0, 0.0, 0.0)
    added_mass_matrix_kg: tuple[tuple[float, ...], ...] | None = None
    # ``None`` preserves the legacy API: non-zero diagonal values imply a
    # diagonal added-mass term. Runtime presets pass ``none``/``diagonal``
    # explicitly, so preset behavior remains unambiguous.
    added_mass_mode: str | None = None
    added_mass_coriolis: bool = False
    preset: Preset = "fast_rl"
    current_velocity_w_mps: Vector3 = (0.0, 0.0, 0.0)

    @property
    def buoyancy_n(self) -> float:
        return self.water_density_kg_m3 * self.displaced_volume_m3 * self.gravity_mps2


class MarineDynamics:
    """Vectorized Fossen terms with a wrench-only external-force interface."""

    def __init__(self, cfg: MarineDynamicsCfg, device: torch.device | str):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = torch.float32
        tensor = lambda value: torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self.center_of_mass = tensor(cfg.center_of_mass_m)
        self.center_of_buoyancy = tensor(cfg.center_of_buoyancy_m)
        self.linear_drag_coeff = tensor(cfg.linear_drag_coeff)
        self.quadratic_drag_coeff = tensor(cfg.quadratic_drag_coeff)
        self.angular_linear_drag_coeff = tensor(cfg.angular_linear_drag_coeff)
        self.angular_quadratic_drag_coeff = tensor(cfg.angular_quadratic_drag_coeff)
        self.added_mass_diag = tensor(cfg.added_mass_kg)
        self.added_inertia_diag = tensor(cfg.added_inertia_kg_m2)
        self.current_velocity_w = tensor(cfg.current_velocity_w_mps)
        self._m_rb = self._build_rigid_body_mass()
        self._m_a = self._build_added_mass()
        self._has_added_mass = bool(torch.any(self._m_a).item())
        validate_symmetric_positive_definite(self._m_rb + self._m_a)

    def _build_rigid_body_mass(self) -> torch.Tensor:
        m = torch.as_tensor(self.cfg.mass_kg, dtype=self.dtype, device=self.device)
        inertia_g = torch.diag(torch.as_tensor(self.cfg.inertia_kg_m2, dtype=self.dtype, device=self.device))
        rg = self.center_of_mass
        inertia_o = inertia_g - m * skew(rg) @ skew(rg)
        return torch.cat(
            (
                torch.cat((m * torch.eye(3, device=self.device), -m * skew(rg)), dim=-1),
                torch.cat((m * skew(rg), inertia_o), dim=-1),
            ),
            dim=-2,
        )

    def _build_added_mass(self) -> torch.Tensor:
        mode = self.cfg.added_mass_mode
        if mode is None:
            has_values = bool(torch.any(self.added_mass_diag).item() or torch.any(self.added_inertia_diag).item())
            mode = "diagonal" if has_values else "none"
        if mode == "none":
            return torch.zeros((6, 6), dtype=self.dtype, device=self.device)
        if self.cfg.added_mass_matrix_kg is not None:
            matrix = torch.as_tensor(self.cfg.added_mass_matrix_kg, dtype=self.dtype, device=self.device)
            if matrix.shape != (6, 6):
                raise ValueError(f"added_mass_matrix_kg must be 6x6, got {tuple(matrix.shape)}")
        else:
            matrix = diagonal_added_mass(self.added_mass_diag, self.added_inertia_diag, self.device, self.dtype)
        if not torch.isfinite(matrix).all():
            raise ValueError("Added-mass matrix contains NaN or Inf")
        if not torch.allclose(matrix, matrix.T, atol=1.0e-6, rtol=1.0e-6):
            raise ValueError("Added-mass matrix must be symmetric")
        if torch.any(torch.linalg.eigvalsh(matrix) < -1.0e-6):
            raise ValueError("Added-mass matrix must be positive semidefinite")
        return matrix

    @property
    def mass_matrix_rb(self) -> torch.Tensor:
        return self._m_rb

    @property
    def mass_matrix_added(self) -> torch.Tensor:
        return self._m_a

    @property
    def mass_matrix(self) -> torch.Tensor:
        return self._m_rb + self._m_a

    def coriolis_rb(self, nu: torch.Tensor) -> torch.Tensor:
        """Rigid-body ``C_RB(nu)`` with ``C + C.T = 0``."""
        nu_1, nu_2 = nu[..., :3], nu[..., 3:]
        m = torch.as_tensor(self.cfg.mass_kg, dtype=nu.dtype, device=nu.device)
        rg = self.center_of_mass.to(dtype=nu.dtype, device=nu.device)
        inertia_g = torch.diag(torch.as_tensor(self.cfg.inertia_kg_m2, dtype=nu.dtype, device=nu.device))
        inertia_o = inertia_g - m * skew(rg) @ skew(rg)
        zero = torch.zeros_like(skew(nu_1))
        top_right = -m * skew(nu_1) - m * skew(nu_2) @ skew(rg)
        bottom_left = -m * skew(nu_1) + m * skew(rg) @ skew(nu_2)
        bottom_right = -skew(torch.matmul(inertia_o, nu_2.unsqueeze(-1)).squeeze(-1))
        return torch.cat((torch.cat((zero, top_right), dim=-1), torch.cat((bottom_left, bottom_right), dim=-1)), dim=-2)

    def coriolis_added(self, nu: torch.Tensor) -> torch.Tensor:
        if not self.cfg.added_mass_coriolis or not self._has_added_mass:
            return torch.zeros(*nu.shape[:-1], 6, 6, dtype=nu.dtype, device=nu.device)
        return added_mass_coriolis(self._m_a.to(dtype=nu.dtype, device=nu.device), nu)

    def coriolis(self, nu: torch.Tensor) -> torch.Tensor:
        return self.coriolis_rb(nu) + self.coriolis_added(nu)

    def damping(self, nu_r: torch.Tensor, scale: torch.Tensor | float | None = None) -> torch.Tensor:
        batch_shape = nu_r.shape[:-1]
        scale_t = as_batch_scale(scale, batch_shape, dtype=nu_r.dtype, device=nu_r.device)
        linear = torch.cat((self.linear_drag_coeff, self.angular_linear_drag_coeff)).to(nu_r) * scale_t.unsqueeze(-1)
        quadratic = torch.cat((self.quadratic_drag_coeff, self.angular_quadratic_drag_coeff)).to(nu_r) * scale_t.unsqueeze(-1)
        return damping_matrix(linear, quadratic, nu_r)

    def restoring_wrench(
        self,
        quat_wxyz: torch.Tensor,
        *,
        buoyancy_scale: torch.Tensor | float | None = None,
        center_of_buoyancy_offset_m: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cob = self.center_of_buoyancy.to(dtype=quat_wxyz.dtype, device=quat_wxyz.device).expand(*quat_wxyz.shape[:-1], 3)
        if center_of_buoyancy_offset_m is not None:
            cob = cob + torch.as_tensor(center_of_buoyancy_offset_m, dtype=quat_wxyz.dtype, device=quat_wxyz.device)
        return buoyancy_wrench_body(quat_wxyz, cob, self.cfg.buoyancy_n, scale=buoyancy_scale)

    def wrench_body(
        self,
        quat_wxyz: torch.Tensor,
        linear_velocity_b: torch.Tensor,
        angular_velocity_b: torch.Tensor,
        *,
        current_velocity_w: torch.Tensor | None = None,
        linear_acceleration_b: torch.Tensor | None = None,
        angular_acceleration_b: torch.Tensor | None = None,
        buoyancy_scale: torch.Tensor | float | None = None,
        drag_scale: torch.Tensor | float | None = None,
        center_of_buoyancy_offset_m: torch.Tensor | None = None,
        added_mass_scale: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_shape = linear_velocity_b.shape[:-1]
        if current_velocity_w is None:
            current_velocity_w = self.current_velocity_w.to(linear_velocity_b).expand_as(linear_velocity_b)
        current_velocity_b = rotate_world_to_body(quat_wxyz, current_velocity_w)
        nu_r = torch.cat((linear_velocity_b - current_velocity_b, angular_velocity_b), dim=-1)
        scale_t = as_batch_scale(added_mass_scale, batch_shape, dtype=linear_velocity_b.dtype, device=linear_velocity_b.device)
        force_b, torque_b = self.restoring_wrench(
            quat_wxyz,
            buoyancy_scale=buoyancy_scale,
            center_of_buoyancy_offset_m=center_of_buoyancy_offset_m,
        )
        damping_force = -torch.matmul(self.damping(nu_r, drag_scale), nu_r.unsqueeze(-1)).squeeze(-1)
        force_b = force_b + damping_force[..., :3]
        torque_b = torque_b + damping_force[..., 3:]
        if self._has_added_mass and (linear_acceleration_b is not None or angular_acceleration_b is not None):
            acceleration = torch.cat(
                (
                    linear_acceleration_b if linear_acceleration_b is not None else torch.zeros_like(linear_velocity_b),
                    angular_acceleration_b if angular_acceleration_b is not None else torch.zeros_like(angular_velocity_b),
                ),
                dim=-1,
            )
            added = torch.matmul(self._m_a.to(linear_velocity_b) * scale_t[..., None, None], acceleration.unsqueeze(-1)).squeeze(-1)
            force_b = force_b - added[..., :3]
            torque_b = torque_b - added[..., 3:]
        if self.cfg.added_mass_coriolis and self._has_added_mass:
            ca_nu = torch.matmul(self.coriolis_added(nu_r), nu_r.unsqueeze(-1)).squeeze(-1)
            force_b = force_b - ca_nu[..., :3]
            torque_b = torque_b - ca_nu[..., 3:]
        return force_b, torque_b
