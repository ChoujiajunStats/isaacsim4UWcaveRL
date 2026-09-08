#!/usr/bin/env python3
"""Headless RGB/depth/IMU/lighting smoke test for the perception mode."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaac_underwater.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg(
        "Isaac-Underwater-PointNav-Perception-v0",
        device=args.device,
        num_envs=args.num_envs,
    )
    env = gym.make("Isaac-Underwater-PointNav-Perception-v0", cfg=cfg)
    env.reset()
    for _ in range(args.steps):
        actions = torch.zeros(args.num_envs, 4, device=env.unwrapped.device)
        _, reward, _, _, _ = env.step(actions)
        assert torch.isfinite(reward).all()
    packets = env.unwrapped.build_sensor_packets()
    assert len(packets) == args.num_envs
    assert all(packet.rgb is not None for packet in packets)
    assert all(packet.depth is not None for packet in packets)
    assert all(packet.imu_acceleration is not None for packet in packets)
    assert all(packet.imu_angular_velocity is not None for packet in packets)
    assert all(packet.ground_truth is not None for packet in packets)
    print(
        f"envs={args.num_envs} steps={args.steps} "
        f"rgb_shape={tuple(packets[0].rgb.shape)} depth_shape={tuple(packets[0].depth.shape)}",
        flush=True,
    )
    print("perception_smoke: PASS", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
