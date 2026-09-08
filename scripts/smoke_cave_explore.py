#!/usr/bin/env python3
"""Validate the no-goal visual exploration environment contract in Isaac."""

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
    task = "Isaac-Underwater-Cave-Explore-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    try:
        obs, _ = env.reset()
        assert set(obs) == {"policy", "critic"}
        assert obs["policy"].shape == (args.num_envs, 1549)
        assert obs["critic"].shape == (args.num_envs, 21)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()

        # A goal mutation must not change either exploration observation.  It
        # is the direct leakage test for route/entrance labels.
        unwrapped = env.unwrapped
        before = unwrapped._get_observations()
        saved_goal = unwrapped._goal_pos_w.clone()
        unwrapped._goal_pos_w.add_(torch.tensor((20.0, -10.0, 5.0), device=unwrapped.device))
        after = unwrapped._get_observations()
        unwrapped._goal_pos_w.copy_(saved_goal)
        torch.testing.assert_close(before["policy"], after["policy"])
        torch.testing.assert_close(before["critic"], after["critic"])

        for _ in range(args.steps):
            obs, reward, terminated, truncated, _ = env.step(
                torch.zeros(args.num_envs, 6, device=unwrapped.device)
            )
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
            assert not torch.any(terminated & truncated)
        tracker = unwrapped._exploration_tracker
        assert tracker is not None
        assert torch.all(tracker.visited_count >= 1)
        assert torch.isfinite(unwrapped._exploration_forward_clearance_m).all()
        assert torch.all(unwrapped._exploration_forward_clearance_m >= 0.249)
        print(
            f"envs={args.num_envs} steps={args.steps} "
            f"policy_shape={tuple(obs['policy'].shape)} critic_shape={tuple(obs['critic'].shape)} "
            f"mean_unique_voxels={tracker.visited_count.float().mean().item():.2f} "
            f"mean_forward_clearance_m={unwrapped._exploration_forward_clearance_m.mean().item():.3f}",
            flush=True,
        )
        print("cave_explore_smoke: PASS", flush=True)
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
