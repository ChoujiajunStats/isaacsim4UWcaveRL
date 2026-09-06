#!/usr/bin/env python3
"""Compare the same wrench trajectory at 1/60, 1/120 and 1/240 seconds."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.validation import make_project_hydro, simulate


def command(_time: float, _nu: torch.Tensor) -> torch.Tensor:
    wrench = torch.zeros(6)
    wrench[0] = 20.0
    wrench[5] = 0.5
    return wrench


def _interp(values: torch.Tensor, source_time: torch.Tensor, target_time: torch.Tensor) -> torch.Tensor:
    indices = torch.searchsorted(source_time, target_time).clamp(1, source_time.numel() - 1)
    left = indices - 1
    right = indices
    weight = ((target_time - source_time[left]) / (source_time[right] - source_time[left])).unsqueeze(-1)
    return values[left] + weight * (values[right] - values[left])


def main() -> None:
    neutral = make_project_hydro(displaced_volume_m3=20.0 / 1025.0)
    dts = (1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0)
    responses = {dt: simulate(neutral, dt=dt, duration_s=4.0, command_wrench_fn=command) for dt in dts}
    reference = responses[dts[-1]]
    rows = []
    for dt in dts:
        response = responses[dt]
        position = _interp(response.pose_w, response.time, reference.time)
        velocity = _interp(response.nu_b, response.time, reference.time)
        position_rmse = torch.sqrt(torch.mean((position - reference.pose_w).square())).item()
        velocity_rmse = torch.sqrt(torch.mean((velocity - reference.nu_b).square())).item()
        quat_dot = (response.quat_wxyz[-1] * reference.quat_wxyz[-1]).sum().abs().clamp(-1.0, 1.0)
        orientation_error = (2.0 * torch.arccos(quat_dot)).item()
        rows.append((dt, position_rmse, velocity_rmse, orientation_error))
    output = PROJECT_ROOT / "benchmarks" / "timestep_convergence.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("physics_dt_s", "position_rmse_m", "velocity_rmse", "orientation_error_rad"))
        writer.writerows(rows)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.loglog([row[0] for row in rows], [row[1] for row in rows], "o-", label="position RMSE")
    axis.loglog([row[0] for row in rows], [row[2] for row in rows], "s-", label="velocity RMSE")
    axis.set_xlabel("physics dt [s]")
    axis.set_ylabel("error vs dt=1/240")
    axis.set_title("Timestep convergence")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(PROJECT_ROOT / "benchmarks" / "plots" / "timestep_convergence.png", dpi=140)
    plt.close(figure)
    for dt, position_rmse, velocity_rmse, orientation_error in rows:
        print(f"dt={dt:.9f} position_rmse={position_rmse:.6e} velocity_rmse={velocity_rmse:.6e} orientation_error={orientation_error:.6e}")
    converges = rows[0][1] > rows[1][1] > rows[2][1] and rows[0][2] > rows[1][2] > rows[2][2]
    print(f"timestep_convergence: {'PASS' if converges else 'FAIL'} output={output}")
    if not converges:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
