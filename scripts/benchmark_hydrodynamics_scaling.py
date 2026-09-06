#!/usr/bin/env python3
"""Launch the physics-only scaling sweep for fast_rl and hydro_rl."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", nargs="+", default=("fast_rl", "hydro_rl"), choices=("fast_rl", "hydro_rl", "reference"))
    parser.add_argument("--envs", nargs="+", type=int, default=(32, 64, 128, 256, 512))
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--measure_steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    python = PROJECT_ROOT / ".venv" / "bin" / "python"
    for preset in args.presets:
        output = PROJECT_ROOT / "logs" / "benchmarks" / f"hydrodynamics_{preset}.csv"
        for count in args.envs:
            command = [
                str(python), str(PROJECT_ROOT / "scripts" / "benchmark.py"),
                "--headless", "--device", args.device,
                "--num_envs", str(count), "--warmup_steps", str(args.warmup_steps),
                "--measure_steps", str(args.measure_steps),
                "--hydrodynamics_preset", preset, "--output", str(output),
            ]
            print("$", " ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
