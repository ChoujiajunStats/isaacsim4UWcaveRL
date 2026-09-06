#!/usr/bin/env python3
"""Run the P0 marine-dynamics response matrix without launching Isaac Sim."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.actuators import ThrusterModel, bluerov2_thruster_layout
from isaac_underwater.validation import Response, make_project_hydro, simulate


def _roll_quat(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0)


def _pitch_quat(angle: float) -> tuple[float, float, float, float]:
    return (math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0)


def _step_wrench(force_axis: int, force_n: float):
    def command(_time: float, _nu: torch.Tensor) -> torch.Tensor:
        value = torch.zeros(6)
        value[force_axis] = force_n
        return value

    return command


def _yaw_step(_time: float, _nu: torch.Tensor) -> torch.Tensor:
    value = torch.zeros(6)
    value[5] = 2.0
    return value


def _smooth_decay(values: torch.Tensor) -> bool:
    magnitudes = values.abs()
    return bool((magnitudes[-1] < magnitudes[0]) and torch.all(magnitudes[1:] <= magnitudes[:-1] + 1.0e-5))


def _write_trajectory(path: Path, trajectories: dict[str, Response]) -> None:
    fields = [
        "experiment", "timestamp", "x", "y", "z", "qw", "qx", "qy", "qz",
        "u", "v", "w", "p", "q", "r",
        *[f"command_{index}" for index in range(8)],
        *[f"thruster_{index}" for index in range(8)],
        "current_x", "current_y", "current_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for name, response in trajectories.items():
            for index in range(response.time.shape[0]):
                row = [name, response.time[index].item()]
                row.extend(response.pose_w[index].tolist())
                row.extend(response.quat_wxyz[index].tolist())
                row.extend(response.nu_b[index].tolist())
                row.extend([0.0] * 8)
                row.extend([0.0] * 8)
                row.extend(response.current_w[index].tolist())
                writer.writerow(row)


def _plot(path: Path, response: Response, component: int, ylabel: str, title: str, *, orientation: bool = False) -> None:
    import matplotlib.pyplot as plt

    if not orientation:
        values = response.nu_b[:, component]
    else:
        q = response.quat_wxyz
        values = torch.atan2(
            2.0 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]),
            1.0 - 2.0 * (q[:, 1].square() + q[:, 2].square()),
        )
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(response.time.numpy(), values.numpy(), linewidth=1.5)
    axis.set_xlabel("time [s]")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    dt = 1.0 / 120.0
    neutral_volume = 20.0 / 1025.0
    neutral = make_project_hydro(displaced_volume_m3=neutral_volume)
    positive = make_project_hydro(displaced_volume_m3=neutral_volume * 1.01)
    trajectories = {
        "surge_step": simulate(neutral, dt=dt, duration_s=4.0, command_wrench_fn=_step_wrench(0, 20.0)),
        "surge_coastdown": simulate(neutral, dt=dt, duration_s=4.0, initial_nu_b=(1.0, 0, 0, 0, 0, 0)),
        "heave_response": simulate(neutral, dt=dt, duration_s=4.0, command_wrench_fn=_step_wrench(2, 8.0)),
        "yaw_response": simulate(neutral, dt=dt, duration_s=4.0, initial_nu_b=(0, 0, 0, 0, 0, 0.8)),
        "restoring_roll": simulate(neutral, dt=dt, duration_s=4.0, initial_quat_wxyz=_roll_quat(math.radians(20.0))),
        "restoring_pitch": simulate(neutral, dt=dt, duration_s=4.0, initial_quat_wxyz=_pitch_quat(math.radians(20.0))),
        "current_response": simulate(neutral, dt=dt, duration_s=4.0, current_w_mps=(0.4, 0.0, 0.0)),
        "positive_buoyancy": simulate(positive, dt=dt, duration_s=4.0),
        "neutral_release": simulate(neutral, dt=dt, duration_s=4.0),
    }
    checks = {
        "T0_neutral_release": trajectories["neutral_release"].pose_w[-1, 2].abs() < 1.0e-4,
        "T1_positive_buoyancy": trajectories["positive_buoyancy"].pose_w[-1, 2] > 0.0,
        "T2_roll_restoring": trajectories["restoring_roll"].nu_b[1, 3] < 0.0,
        "T3_pitch_restoring": trajectories["restoring_pitch"].nu_b[1, 4] < 0.0,
        "T4_surge_coastdown": _smooth_decay(trajectories["surge_coastdown"].nu_b[:, 0]),
        "T5_sway_coastdown": True,
        "T6_heave_coastdown": True,
        "T7_yaw_decay": _smooth_decay(trajectories["yaw_response"].nu_b[:, 5]),
        "T8_relative_current": trajectories["current_response"].nu_b[-1, 0] > 0.0,
    }
    # Run sway/heave coast-down and keep their traces in the same trajectory file.
    sway = simulate(neutral, dt=dt, duration_s=4.0, initial_nu_b=(0, 1.0, 0, 0, 0, 0))
    heave = simulate(neutral, dt=dt, duration_s=4.0, initial_nu_b=(0, 0, 1.0, 0, 0, 0))
    trajectories["sway_coastdown"] = sway
    trajectories["heave_coastdown"] = heave
    checks["T5_sway_coastdown"] = _smooth_decay(sway.nu_b[:, 1])
    checks["T6_heave_coastdown"] = _smooth_decay(heave.nu_b[:, 2])

    thrusters = ThrusterModel(bluerov2_thruster_layout(), "cpu")
    thrusters.reset(8)
    commands = torch.eye(8)
    thrust = thrusters.command_to_target_thrust(commands)
    force, torque = thrusters.wrench_from_thrust(thrust)
    thruster_checks = []
    for index, config in enumerate(bluerov2_thruster_layout()):
        direction = torch.tensor(config.direction_b)
        thruster_checks.append(bool(torch.dot(force[index], direction) > 0.0))
        print(f"T9_thruster_{index}: {'PASS' if thruster_checks[-1] else 'FAIL'} force={force[index].tolist()} torque={torque[index].tolist()}")
    checks["T9_individual_thrusters"] = all(thruster_checks)

    output_dir = PROJECT_ROOT / "benchmarks"
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _write_trajectory(output_dir / "hydro_trajectories.csv", trajectories)
    _plot(plot_dir / "surge_step.png", trajectories["surge_step"], 0, "surge velocity [m/s]", "Surge step")
    _plot(plot_dir / "surge_coastdown.png", trajectories["surge_coastdown"], 0, "surge velocity [m/s]", "Surge coast-down")
    _plot(plot_dir / "heave_response.png", trajectories["heave_response"], 2, "heave velocity [m/s]", "Heave response")
    _plot(plot_dir / "yaw_response.png", trajectories["yaw_response"], 5, "yaw rate [rad/s]", "Yaw decay")
    _plot(plot_dir / "restoring_roll.png", trajectories["restoring_roll"], 0, "roll angle [rad]", "Roll restoring", orientation=True)
    _plot(plot_dir / "current_response.png", trajectories["current_response"], 0, "surge velocity [m/s]", "Current-relative response")
    status = "PASS" if all(checks.values()) else "FAIL"
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"status={status} trajectory={output_dir / 'hydro_trajectories.csv'} plots={plot_dir}")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
