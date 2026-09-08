#!/usr/bin/env python3
"""Validate heterogeneous cave spawning and entrance-to-exit task semantics."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=3)
parser.add_argument("--profile", default="train_all")
parser.add_argument("--domain_randomization", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

os.environ["ISAAC_UNDERWATER_CAVE_PROFILE"] = args.profile
if args.domain_randomization:
    os.environ["ISAAC_UNDERWATER_MULTICAVE_DOMAIN_RANDOMIZATION"] = "1"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import omni.usd
import torch

import isaac_underwater.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    task = "Isaac-Underwater-Cave-Navigation-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    env = gym.make(task, cfg=cfg)
    try:
        obs, _ = env.reset()
        unwrapped = env.unwrapped
        variants = tuple(cfg.world.scene_variants)
        assert obs["policy"].shape == (args.num_envs, 1553)
        assert obs["critic"].shape == (args.num_envs, 19)
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(obs["critic"]).all()
        assert len(variants) <= args.num_envs
        expected_assignment = torch.arange(args.num_envs, device=unwrapped.device) % len(variants)
        torch.testing.assert_close(unwrapped._cave_scene_ids, expected_assignment)

        relative_spawn = unwrapped._robot.data.root_pos_w - unwrapped.scene.env_origins
        expected_spawn = unwrapped._cave_scene_spawn_points[unwrapped._cave_scene_ids]
        spawn_error = torch.linalg.vector_norm(relative_spawn - expected_spawn, dim=-1)
        assert torch.all(spawn_error < 0.75), f"spawn error too large: {spawn_error}"
        assert torch.all(unwrapped._cave_scene_route_lengths > 1.0)
        if args.domain_randomization:
            assert unwrapped._domain_gap_samples is not None
            # Deterministically exercise the fixed-tensor frame-hold path.
            unwrapped._domain_gap_samples["frame_drop_probability"].fill_(1.0)

        stage = omni.usd.get_context().get_stage()
        for env_index in range(args.num_envs):
            visual = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Cave")
            assert visual.IsValid()
            assert stage.GetPrimAtPath(f"/World/envs/env_{env_index}/CaveCollision").IsValid()
            scene = variants[int(unwrapped._cave_scene_ids[env_index])]
            references = visual.GetMetadata("references").GetAddedOrExplicitItems()
            referenced_paths = {
                str(Path(reference.assetPath).expanduser().resolve())
                for reference in references
                if reference.assetPath
            }
            assert str(Path(scene.visual_asset_path).resolve()) in referenced_paths, (
                f"env_{env_index} does not reference selected scene {scene.key}: {referenced_paths}"
            )

        held_obs, _, initial_terminated, initial_truncated, initial_extras = env.step(
            torch.zeros(args.num_envs, 6, device=unwrapped.device)
        )
        assert torch.isfinite(held_obs["policy"]).all()
        assert not torch.any(initial_terminated | initial_truncated), (
            "entrance spawn terminated immediately: "
            f"success={initial_extras['episode_success'].tolist()} "
            f"bounds={initial_extras['episode_out_of_bounds'].tolist()} "
            f"collision={initial_extras['episode_collision'].tolist()}"
        )
        assert not torch.any(unwrapped._cave_collision), (
            f"entrance spawn starts in collision: {unwrapped._cave_contact_force_n}"
        )

        # Force one exact exit event per scene.  This checks task success,
        # immediate-reset outcome preservation, path metrics, and SPL without
        # claiming anything about an untrained policy.
        goal_local = unwrapped._cave_scene_goal_points[unwrapped._cave_scene_ids]
        pose = unwrapped._robot.data.root_state_w[:, :7].clone()
        pose[:, :3] = goal_local + unwrapped.scene.env_origins
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
        assert torch.all(extras["episode_reference_path_m"] > 1.0)
        assert torch.all((extras["episode_spl"] > 0.0) & (extras["episode_spl"] <= 1.0))
        assert torch.all(reward > 0.25 * float(cfg.goal_bonus))

        scene_summary = ",".join(
            f"{scene.key}:{int((unwrapped._cave_scene_ids == index).sum())}"
            for index, scene in enumerate(variants)
        )
        print(
            f"profile={cfg.world.dataset_profile} assignment={scene_summary} "
            f"policy_shape={tuple(obs['policy'].shape)} critic_shape={tuple(obs['critic'].shape)} "
            f"route_lengths_m={[round(value, 2) for value in unwrapped._cave_scene_route_lengths.tolist()]}",
            flush=True,
        )
        print("cave_navigation_smoke: PASS", flush=True)
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
