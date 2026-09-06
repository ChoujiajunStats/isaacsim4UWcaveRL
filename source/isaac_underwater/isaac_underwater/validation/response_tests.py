"""Small Fossen ODE harness used for P0 response and timestep tests.

This is deliberately separate from the Isaac task.  It validates the wrench
terms and numerical behavior; the live task delegates integration to PhysX.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT / "source" / "isaac_underwater") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.config import load_config
from isaac_underwater.physics import Hydrodynamics, HydrodynamicsCfg, rotate_body_to_world, rotate_world_to_body


@dataclass
class Response:
    time: torch.Tensor
    pose_w: torch.Tensor
    quat_wxyz: torch.Tensor
    nu_b: torch.Tensor
    command_wrench_b: torch.Tensor
    current_w: torch.Tensor


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _integrate_quaternion(quat: torch.Tensor, omega_b: torch.Tensor, dt: float) -> torch.Tensor:
    omega_quat = torch.cat((torch.zeros_like(omega_b[..., :1]), omega_b), dim=-1)
    updated = quat + 0.5 * _quat_multiply(quat, omega_quat) * dt
    return updated / torch.linalg.vector_norm(updated, dim=-1, keepdim=True).clamp_min(1.0e-8)


def make_project_hydro(*, displaced_volume_m3: float | None = None) -> Hydrodynamics:
    """Build a project model from the provenance-aware YAML contract."""
    contract = load_config("robots/bluerov2_hydro.yaml")
    linear_damping = contract["linear_damping"]
    quadratic_damping = contract["quadratic_damping"]
    return Hydrodynamics(
        HydrodynamicsCfg(
            water_density_kg_m3=contract["water"]["density"]["nominal"],
            displaced_volume_m3=contract["volume"]["nominal"] if displaced_volume_m3 is None else displaced_volume_m3,
            gravity_mps2=contract["water"]["gravity"]["nominal"],
            mass_kg=contract["mass"]["nominal"],
            inertia_kg_m2=tuple(contract["inertia"]["nominal"]),
            center_of_mass_m=tuple(contract["center_of_mass"]["nominal"]),
            center_of_buoyancy_m=tuple(contract["center_of_buoyancy"]["nominal"]),
            linear_drag_coeff=tuple(linear_damping[axis]["nominal"] for axis in ("surge", "sway", "heave")),
            quadratic_drag_coeff=tuple(quadratic_damping[axis]["nominal"] for axis in ("surge", "sway", "heave")),
            angular_linear_drag_coeff=tuple(linear_damping[axis]["nominal"] for axis in ("roll", "pitch", "yaw")),
            angular_quadratic_drag_coeff=tuple(quadratic_damping[axis]["nominal"] for axis in ("roll", "pitch", "yaw")),
            preset="fast_rl",
            added_mass_mode="none",
        ),
        "cpu",
    )


def simulate(
    hydro: Hydrodynamics,
    *,
    dt: float,
    duration_s: float,
    initial_nu_b: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    initial_quat_wxyz: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0),
    current_w_mps: tuple[float, ...] = (0.0, 0.0, 0.0),
    command_wrench_fn=None,
) -> Response:
    """Integrate the Fossen equation for response testing only."""
    steps = int(round(duration_s / dt)) + 1
    time = torch.arange(steps, dtype=torch.float32) * dt
    pose = torch.zeros(steps, 3)
    quat = torch.zeros(steps, 4)
    nu = torch.zeros(steps, 6)
    command_log = torch.zeros(steps, 6)
    current_log = torch.zeros(steps, 3)
    quat[0] = torch.tensor(initial_quat_wxyz)
    nu[0] = torch.tensor(initial_nu_b)
    current = torch.tensor(current_w_mps, dtype=torch.float32)
    mass = hydro.cfg.mass_kg
    gravity_w = torch.tensor([0.0, 0.0, -mass * hydro.cfg.gravity_mps2])
    mass_matrix = hydro.mass_matrix
    for index in range(steps - 1):
        q = quat[index : index + 1]
        velocity = nu[index : index + 1]
        current_batch = current.reshape(1, 3)
        current_body = rotate_world_to_body(q, current_batch)
        nu_r = velocity.clone()
        nu_r[:, :3] -= current_body
        command = torch.zeros(1, 6) if command_wrench_fn is None else torch.as_tensor(
            command_wrench_fn(float(time[index]), velocity[0]), dtype=torch.float32
        ).reshape(1, 6)
        hydro_force, hydro_torque = hydro.wrench_body(
            q,
            velocity[:, :3],
            velocity[:, 3:],
            current_velocity_w=current_batch,
        )
        weight_b = rotate_world_to_body(q, gravity_w.reshape(1, 3))
        net = torch.cat((hydro_force + weight_b, hydro_torque), dim=-1) + command
        coriolis = torch.matmul(hydro.coriolis(nu_r), nu_r.unsqueeze(-1)).squeeze(-1)
        acceleration = torch.linalg.solve(mass_matrix, (net - coriolis).T).T
        nu[index + 1] = velocity[0] + dt * acceleration[0]
        pose[index + 1] = pose[index] + dt * rotate_body_to_world(q, velocity[:, :3])[0]
        quat[index + 1] = _integrate_quaternion(q, velocity[:, 3:], dt)[0]
        command_log[index] = command[0]
        current_log[index] = current
    command_log[-1] = command_log[-2]
    current_log[-1] = current
    return Response(time, pose, quat, nu, command_log, current_log)
