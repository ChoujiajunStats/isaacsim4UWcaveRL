#!/usr/bin/env python3
"""Headless stereo RGB/depth and per-environment active-light smoke test."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import omni.usd

import isaac_underwater.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = "Isaac-Underwater-PointNav-Stereo-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    env.reset()
    actions = torch.zeros(args.num_envs, 6, device=env.unwrapped.device)
    actions[:, 4] = -1.0  # left lamp off
    actions[:, 5] = 1.0  # right lamp full scale
    if args.num_envs > 1:
        actions[1:, 4] = 1.0  # reverse the lamp schedule in the second envs
        actions[1:, 5] = -1.0
    for _ in range(args.steps):
        _, reward, _, _, _ = env.step(actions)
        assert torch.isfinite(reward).all()
    packets = env.unwrapped.build_sensor_packets()
    assert len(packets) == args.num_envs
    assert all(packet.rgb_left is not None and packet.rgb_right is not None for packet in packets)
    assert all(packet.depth_left is not None and packet.depth_right is not None for packet in packets)
    assert all(packet.rgb is not None and packet.depth is not None for packet in packets)
    assert all(packet.sensor_metadata["stereo"] for packet in packets)
    assert all(abs(packet.sensor_metadata["stereo_baseline_m"] - 0.145) < 1.0e-6 for packet in packets)
    assert all(abs(packet.sensor_metadata["stereo_near_clip_m"] - 0.25) < 1.0e-6 for packet in packets)
    assert all(torch.isfinite(packet.depth.float()).any() for packet in packets)
    assert all(packet.rgb.float().std() > 0.0 for packet in packets)
    stage = omni.usd.get_context().get_stage()
    left = stage.GetPrimAtPath("/World/envs/env_0/Robot/FrontLightLeft")
    right = stage.GetPrimAtPath("/World/envs/env_0/Robot/FrontLightRight")
    assert left.IsValid() and right.IsValid()
    def intensity(prim):
        for name in ("inputs:intensity", "intensity"):
            attr = prim.GetAttribute(name)
            if attr.IsValid():
                return attr.Get()
        raise AssertionError(f"No intensity attribute on {prim.GetPath()}")

    left_intensity = intensity(left)
    right_intensity = intensity(right)
    assert left_intensity < right_intensity
    if args.num_envs > 1:
        left_1 = stage.GetPrimAtPath("/World/envs/env_1/Robot/FrontLightLeft")
        right_1 = stage.GetPrimAtPath("/World/envs/env_1/Robot/FrontLightRight")
        assert intensity(left_1) > intensity(right_1)
    print(
        f"envs={args.num_envs} steps={args.steps} "
        f"left_rgb={tuple(packets[0].rgb_left.shape)} right_rgb={tuple(packets[0].rgb_right.shape)} "
        f"baseline_m={packets[0].sensor_metadata['stereo_baseline_m']}",
        flush=True,
    )
    print("stereo_lighting_smoke: PASS", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
