#!/usr/bin/env python3
"""Monte-Carlo and analytical checks for the Fossen-compatible terms."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.config import load_config
from isaac_underwater.physics import Hydrodynamics, HydrodynamicsCfg


def _quat_from_rpy(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _make_model() -> Hydrodynamics:
    contract = load_config("robots/bluerov2_hydro.yaml")
    added = contract["added_mass"]
    preset = contract["presets"]["hydro_rl"]
    drag = load_config("underwater.yaml")["drag"]
    return Hydrodynamics(
        HydrodynamicsCfg(
            water_density_kg_m3=contract["water"]["density"]["nominal"],
            displaced_volume_m3=contract["volume"]["nominal"],
            gravity_mps2=contract["water"]["gravity"]["nominal"],
            mass_kg=contract["mass"]["nominal"],
            inertia_kg_m2=tuple(contract["inertia"]["nominal"]),
            center_of_mass_m=tuple(contract["center_of_mass"]["nominal"]),
            center_of_buoyancy_m=tuple(contract["center_of_buoyancy"]["nominal"]),
            linear_drag_coeff=tuple(drag["linear_coeff"]),
            quadratic_drag_coeff=tuple(drag["quadratic_coeff"]),
            angular_linear_drag_coeff=tuple(drag["angular_linear_coeff"]),
            angular_quadratic_drag_coeff=tuple(drag["angular_quadratic_coeff"]),
            added_mass_kg=tuple(added["translational"]["nominal"]),
            added_inertia_kg_m2=tuple(added["rotational"]["nominal"]),
            added_mass_mode=preset["added_mass_mode"],
            added_mass_coriolis=preset["added_mass_coriolis"],
            preset="hydro_rl",
        ),
        "cpu",
    )


def main() -> None:
    count = 10000
    torch.manual_seed(7)
    hydro = _make_model()
    m = hydro.mass_matrix
    symmetry_error = (m - m.T).abs().max().item()
    eigenvalues = torch.linalg.eigvalsh(m)
    nu = torch.randn(count, 6)
    c = hydro.coriolis(nu)
    coriolis_energy = torch.einsum("bi,bij,bj->b", nu, c, nu).abs().max().item()
    damping_matrix = hydro.damping(nu)
    damping_power = torch.einsum("bi,bij,bj->b", nu, damping_matrix, nu)
    quat = _quat_from_rpy(torch.randn(count) * math.pi, torch.randn(count) * 0.5, torch.randn(count) * math.pi)
    force, torque = hydro.wrench_body(quat, nu[:, :3], nu[:, 3:], current_velocity_w=torch.randn(count, 3) * 0.3)
    relative = torch.cat((nu[:, :3], nu[:, 3:]), dim=-1)
    damping_wrench = -torch.matmul(damping_matrix, relative.unsqueeze(-1)).squeeze(-1)
    wrench_power = torch.sum(damping_wrench * relative, dim=-1)

    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    zero = torch.zeros(1, 3)
    same_water = torch.tensor([[1.0, 0.0, 0.0]])
    same_force, _ = hydro.wrench_body(identity_quat, same_water, zero, current_velocity_w=same_water)
    current_force, _ = hydro.wrench_body(identity_quat, zero, zero, current_velocity_w=same_water)
    roll = math.radians(20.0)
    roll_q = torch.tensor([[math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]])
    _, roll_torque = hydro.wrench_body(roll_q, zero, zero)
    pitch = math.radians(20.0)
    pitch_q = torch.tensor([[math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]])
    _, pitch_torque = hydro.wrench_body(pitch_q, zero, zero)

    checks = {
        "mass_matrix_symmetry": symmetry_error <= 1.0e-6,
        "mass_matrix_positive_definite": bool(torch.all(eigenvalues > 0.0)),
        "coriolis_energy_property": coriolis_energy <= 1.0e-4,
        "damping_dissipation_matrix": bool(torch.all(damping_power >= -1.0e-6)),
        "damping_wrench_dissipation": bool(torch.all(wrench_power <= 1.0e-4)),
        "finite_values": bool(torch.isfinite(force).all() and torch.isfinite(torque).all()),
        "zero_relative_current_drag": bool(same_force[0, :2].abs().max() < 1.0e-5),
        "stationary_current_force_direction": bool(current_force[0, 0] > 0.0),
        "roll_restoring_direction": bool(roll_torque[0, 0] * roll < 0.0),
        "pitch_restoring_direction": bool(pitch_torque[0, 1] * pitch < 0.0),
    }
    results = {
        "samples": count,
        "preset": "hydro_rl",
        "mass_eigenvalues": eigenvalues.tolist(),
        "max_mass_symmetry_error": symmetry_error,
        "max_coriolis_energy_abs": coriolis_energy,
        "min_damping_power": damping_power.min().item(),
        "max_wrench_power": wrench_power.max().item(),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    out = PROJECT_ROOT / "benchmarks" / "analytical_invariants.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"samples={count} status={results['status']} output={out}")
    if results["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
