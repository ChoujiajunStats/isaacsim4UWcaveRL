#!/usr/bin/env python3
"""Measure physics-only vectorized PointNav throughput and VRAM."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--mode", choices=("physics", "perception"), default="physics")
parser.add_argument("--warmup_steps", type=int, default=100)
parser.add_argument("--measure_steps", type=int, default=500)
parser.add_argument("--hydrodynamics_preset", choices=("fast_rl", "hydro_rl", "reference"), default="fast_rl")
parser.add_argument("--output", type=Path, default=Path("logs/benchmarks/physics.csv"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.mode == "perception":
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaac_underwater.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def gpu_metrics() -> tuple[int, int]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    memory_mb, utilization = output.split(",", maxsplit=1)
    return int(memory_mb.strip()), int(utilization.strip())


def append_result(path: Path, result: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result))
        if write_header:
            writer.writeheader()
        writer.writerow(result)


def main() -> None:
    task_name = (
        "Isaac-Underwater-PointNav-Perception-v0"
        if args.mode == "perception"
        else "Isaac-Underwater-PointNav-Direct-v0"
    )
    cfg = parse_env_cfg(
        task_name,
        device=args.device,
        num_envs=args.num_envs,
    )
    cfg.hydrodynamics_preset = args.hydrodynamics_preset
    env = gym.make(task_name, cfg=cfg)
    env.reset()
    actions = torch.zeros(args.num_envs, 4, device=env.unwrapped.device)
    for _ in range(args.warmup_steps):
        env.step(actions)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.measure_steps):
        env.step(actions)
        if args.mode == "perception":
            packets = env.unwrapped.build_sensor_packets()
            if not packets or packets[0].rgb is None or packets[0].depth is None:
                raise RuntimeError("Perception benchmark did not produce RGB and depth outputs")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    vram_mb, gpu_utilization = gpu_metrics()
    result = {
        "mode": args.mode,
        "hydrodynamics_preset": args.hydrodynamics_preset,
        "num_envs": args.num_envs,
        "control_steps": args.measure_steps,
        "elapsed_s": round(elapsed, 6),
        "env_steps_per_s": round(args.num_envs * args.measure_steps / elapsed, 2),
        "sim_steps_per_s": round(args.measure_steps * cfg.decimation / elapsed, 2),
        "real_time_factor": round(args.measure_steps * env.unwrapped.step_dt / elapsed, 3),
        "vram_mb": vram_mb,
        "gpu_utilization_pct": gpu_utilization,
        "device": str(env.unwrapped.device),
    }
    append_result(args.output, result)
    print(result)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
