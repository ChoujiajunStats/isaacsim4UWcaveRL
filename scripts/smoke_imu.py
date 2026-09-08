#!/usr/bin/env python3
"""Camera-free IMU and vehicle-light smoke test."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--disable_lighting", action="store_true")
parser.add_argument("--domain_randomization", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import omni.usd

import isaac_underwater.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = "Isaac-Underwater-PointNav-IMU-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = 42
    if args.disable_lighting:
        cfg.lighting.enabled = False
    cfg.domain_randomization_enabled = args.domain_randomization
    env = gym.make(task, cfg=cfg)
    assert env.unwrapped._camera is None
    assert env.unwrapped._imu is not None
    env.reset()
    for _ in range(args.steps):
        _, reward, _, _, _ = env.step(torch.zeros(args.num_envs, 4, device=env.unwrapped.device))
        assert torch.isfinite(reward).all()
    packets = env.unwrapped.build_sensor_packets()
    assert len(packets) == args.num_envs
    assert all(packet.imu_acceleration is not None for packet in packets)
    assert all(packet.imu_angular_velocity is not None for packet in packets)
    if args.domain_randomization:
        assert all("camera_imu_offset_s" in (packet.sensor_metadata or {}) for packet in packets)
        assert all("extrinsic_rotation_deg" in (packet.sensor_metadata or {}) for packet in packets)
    stage = omni.usd.get_context().get_stage()
    assert stage.GetPrimAtPath("/World/UnderwaterAmbient").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_0/Robot/FrontLightLeft").IsValid()
    assert stage.GetPrimAtPath("/World/envs/env_0/Robot/FrontLightRight").IsValid()
    print(
        f"envs={args.num_envs} steps={args.steps} imu_shape={tuple(packets[0].imu_acceleration.shape)}",
        flush=True,
    )
    print("imu_lighting_smoke: PASS", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
