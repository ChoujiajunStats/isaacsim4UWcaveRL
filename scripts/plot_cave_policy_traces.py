#!/usr/bin/env python3
"""Plot recorded navigation diagnostic trajectories; no simulator required."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = json.loads(args.trace.read_text(encoding="utf-8"))
    scenes = list(report["phases"]["full_route"])
    figure, axes = plt.subplots(2, len(scenes), figsize=(5 * len(scenes), 8), squeeze=False)
    colors = {"full_route": "#168a9c", "curriculum_frontier": "#e58922"}
    labels = {"full_route": "PPO from entrance", "curriculum_frontier": "PPO from saved frontier"}
    for column, name in enumerate(scenes):
        map_axis, deviation_axis = axes[:, column]
        record = report["phases"]["full_route"][name]
        route = np.array(record["reference_route_m"])
        goal = np.array(record["goal_m"])
        map_axis.plot(route[:, 0], route[:, 1], color="#999999", linewidth=3, alpha=.6, label="Reference route")
        map_axis.scatter(goal[0], goal[1], s=150, marker="*", c="#248043", label="Exit", zorder=5)
        for phase in colors:
            record = report["phases"][phase][name]
            samples = record["samples"]
            positions = np.array([sample["position_m"] for sample in samples])
            map_axis.plot(positions[:, 0], positions[:, 1], color=colors[phase], label=labels[phase])
            map_axis.scatter(positions[0, 0], positions[0, 1], color=colors[phase], marker="o", s=35, zorder=5)
            map_axis.scatter(positions[-1, 0], positions[-1, 1], color=colors[phase], marker="x", s=100, zorder=6)
            deviation_axis.plot([sample["step"] * .05 for sample in samples],
                                [sample["route_deviation_m"] for sample in samples],
                                color=colors[phase], label=labels[phase])
        map_axis.set(title=f"{name}: XY projection (x = failure)", xlabel="x [m]", ylabel="y [m]")
        map_axis.set_aspect("equal", adjustable="datalim")
        map_axis.grid(alpha=.2)
        deviation_axis.axhline(2.25, color="#b02b30", linestyle="--", label="Route-bounds limit")
        deviation_axis.set(xlabel="Episode time [s]", ylabel="3D distance from route [m]", ylim=(0, 2.6))
        deviation_axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    figure.suptitle("PPO checkpoint 607: one seeded episode per cave and start condition\n"
                   "Diagnostic only; reference route is NOT a collision mesh or actor input", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .93))
    output = args.output or args.trace.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
