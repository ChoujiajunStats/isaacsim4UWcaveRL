#!/usr/bin/env python3
"""Validate the visual actor / privileged critic PPO environment contract."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3)
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
    task = "Isaac-Underwater-Cave-VisualPilot-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    try:
        obs, _ = env.reset()
        assert set(obs) == {"policy", "critic"}
        assert obs["policy"].shape == (args.num_envs, 1553)
        assert obs["critic"].shape == (args.num_envs, 19)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(obs["critic"]).all()
        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(
                torch.zeros(args.num_envs, 6, device=env.unwrapped.device)
            )
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
            assert not torch.any(terminated & truncated)
        unwrapped = env.unwrapped
        assert torch.isfinite(unwrapped._cave_centerline_distance).all()
        assert torch.isfinite(unwrapped._cave_route_chainage).all()
        assert torch.isfinite(unwrapped._cave_contact_force_n).all()
        print(
            f"envs={args.num_envs} steps={args.steps} "
            f"policy_shape={tuple(obs['policy'].shape)} critic_shape={tuple(obs['critic'].shape)} "
            f"route_deviation_m={unwrapped._cave_centerline_distance.mean().item():.3f} "
            f"contact_force_n={unwrapped._cave_contact_force_n.mean().item():.3f}",
            flush=True,
        )
        print("visual_pilot_smoke: PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
