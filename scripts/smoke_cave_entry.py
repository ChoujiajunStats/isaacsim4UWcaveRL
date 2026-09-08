#!/usr/bin/env python3
"""Validate automatic portal inference and the vision-only entry task."""

from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
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
    task = "Isaac-Underwater-Cave-Entry-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    try:
        obs, _ = env.reset()
        unwrapped = env.unwrapped
        assert obs["policy"].shape == (args.num_envs, 1549)
        assert obs["critic"].shape == (args.num_envs, 24)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
        assert len(unwrapped._cave_portals) == 1
        portal = unwrapped._cave_portals[0]
        assert portal.endpoint == "start"
        assert portal.exterior_run_m >= float(cfg.cave_entry_minimum_exterior_run_m)
        assert portal.local_clearance_m >= float(cfg.cave_entry_minimum_clearance_m)
        assert torch.all(unwrapped._cave_entry_signed_depth_m < 0.0)
        assert not torch.any(unwrapped._episode_entered_cave)

        # Neither the actor nor privileged critic may receive the legacy goal.
        before = unwrapped._get_observations()
        saved_goal = unwrapped._goal_pos_w.clone()
        unwrapped._goal_pos_w.add_(torch.tensor((30.0, -20.0, 4.0), device=unwrapped.device))
        after = unwrapped._get_observations()
        unwrapped._goal_pos_w.copy_(saved_goal)
        torch.testing.assert_close(before["policy"], after["policy"])
        torch.testing.assert_close(before["critic"], after["critic"])

        # Put each robot just through its inferred portal.  The following real
        # environment step must emit the sparse entry event, terminate, and
        # preserve success through DirectRLEnv's immediate reset.
        portal_ids = unwrapped._cave_entry_portal_id
        target_local = unwrapped._cave_portal_positions[portal_ids] + (
            unwrapped._cave_portal_directions[portal_ids]
            * (float(cfg.cave_entry_depth_m) + 0.10)
        )
        pose = unwrapped._robot.data.root_state_w[:, :7].clone()
        pose[:, :3] = target_local + unwrapped.scene.env_origins
        unwrapped._robot.write_root_pose_to_sim(pose)
        unwrapped._robot.write_root_velocity_to_sim(
            torch.zeros(args.num_envs, 6, device=unwrapped.device)
        )
        _, reward, terminated, truncated, extras = env.step(
            torch.zeros(args.num_envs, 6, device=unwrapped.device)
        )
        assert torch.all(terminated)
        assert not torch.any(truncated)
        assert torch.all(extras["episode_success"])
        assert torch.all(reward > 0.5 * float(cfg.cave_entry_bonus))
        print(
            f"portal_source=automatic_clearance_transition endpoint={portal.endpoint} "
            f"chainage_m={portal.chainage_m:.2f} exterior_run_m={portal.exterior_run_m:.2f} "
            f"local_clearance_m={portal.local_clearance_m:.3f} "
            f"policy_shape={tuple(obs['policy'].shape)} critic_shape={tuple(obs['critic'].shape)}",
            flush=True,
        )
        print("cave_entry_smoke: PASS", flush=True)
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
