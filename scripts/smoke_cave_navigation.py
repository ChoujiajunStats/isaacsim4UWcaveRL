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
parser.add_argument("--navigation_curriculum", action="store_true")
parser.add_argument(
    "--collision_reset_steps",
    type=int,
    default=0,
    help="Drive the hard-scene robot sideways and verify its post-contact reset.",
)
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
from isaac_underwater.navigation.route_geometry import sample_polyline


def main() -> None:
    task = "Isaac-Underwater-Cave-Navigation-v0"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.navigation_curriculum_enabled = args.navigation_curriculum
    if args.collision_reset_steps > 0:
        # Let the robot reach the physical wall before the route-deviation
        # safety bound terminates the diagnostic episode.
        cfg.route_deviation_hard_m = 100.0
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
        if args.navigation_curriculum:
            ids = unwrapped._cave_scene_ids
            expected_spawn, _ = sample_polyline(
                unwrapped._cave_scene_goal_chainages[ids] - unwrapped._exit_curriculum.distance_m[ids],
                unwrapped._multi_cave_centerlines[ids],
                unwrapped._multi_cave_chainages[ids],
                unwrapped._multi_cave_route_mask[ids],
            )
            assert torch.all(unwrapped._episode_route_reference_m <= cfg.navigation_curriculum_initial_distance_m + 1.e-4)
            assert torch.all(unwrapped._episode_navigation_horizon_steps < unwrapped.max_episode_length)
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
        assert torch.all(
            unwrapped._cave_contact_force_n <= float(cfg.collision_force_threshold_n)
        ), f"entrance spawn has raw contact force: {unwrapped._cave_contact_force_n}"

        if args.collision_reset_steps > 0:
            hard_scene_ids = [
                index for index, scene in enumerate(variants) if scene.difficulty == "hard"
            ]
            if len(hard_scene_ids) != 1:
                raise ValueError("--collision_reset_steps requires exactly one hard scene")
            hard_env_ids = (unwrapped._cave_scene_ids == hard_scene_ids[0]).nonzero(
                as_tuple=False
            ).flatten()
            if hard_env_ids.numel() == 0:
                raise AssertionError("No environment was assigned to the hard scene")
            hard_env_id = int(hard_env_ids[0].item())
            # Isolate reset/contact behavior from the ordinary spawn jitter.
            unwrapped._cave_scene_spawn_jitters[hard_scene_ids[0]] = 0.0
            route_mask = unwrapped._multi_cave_route_mask[hard_scene_ids[0]]
            route = unwrapped._multi_cave_centerlines[hard_scene_ids[0], route_mask]
            chainage = unwrapped._multi_cave_chainages[hard_scene_ids[0], route_mask]
            interior_chainage = unwrapped._cave_scene_spawn_chainages[hard_scene_ids[0]] + 10.0
            interior_index = int(torch.argmin((chainage - interior_chainage).abs()).item())
            tangent_index = min(interior_index + 1, route.shape[0] - 1)
            tangent = route[tangent_index] - route[interior_index]
            yaw = torch.atan2(tangent[1], tangent[0])
            pose = unwrapped._robot.data.root_state_w[hard_env_id : hard_env_id + 1, :7].clone()
            pose[:, :3] = route[interior_index] + unwrapped.scene.env_origins[hard_env_id]
            pose[:, 3:7] = 0.0
            pose[:, 3] = torch.cos(0.5 * yaw)
            pose[:, 6] = torch.sin(0.5 * yaw)
            unwrapped._robot.write_root_pose_to_sim(pose, hard_env_ids[:1])
            unwrapped._robot.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=unwrapped.device), hard_env_ids[:1]
            )
            collision_seen = False
            for _ in range(args.collision_reset_steps):
                drive = torch.zeros(args.num_envs, 6, device=unwrapped.device)
                drive[hard_env_id, 1] = 1.0
                _, _, terminated, truncated, extras = env.step(drive)
                done = terminated | truncated
                if bool(done[hard_env_id] and extras["episode_collision"][hard_env_id]):
                    collision_seen = True
                    post_reset_steps = 3
                    post_reset_trace = []
                    for post_reset_step in range(1, post_reset_steps + 1):
                        _, _, next_terminated, next_truncated, next_extras = env.step(
                            torch.zeros(args.num_envs, 6, device=unwrapped.device)
                        )
                        reset_position = (
                            unwrapped._robot.data.root_pos_w[hard_env_id]
                            - unwrapped.scene.env_origins[hard_env_id]
                        )
                        post_reset_trace.append(
                            (
                                post_reset_step,
                                [round(float(value), 3) for value in reset_position],
                                round(float(unwrapped._cave_contact_force_n[hard_env_id]), 3),
                            )
                        )
                        assert not bool(
                            next_terminated[hard_env_id] | next_truncated[hard_env_id]
                        ), (
                            "hard scene terminated again after contact reset "
                            f"at step {post_reset_step}: "
                            f"collision={bool(next_extras['episode_collision'][hard_env_id])} "
                            f"bounds={bool(next_extras['episode_out_of_bounds'][hard_env_id])} "
                            f"success={bool(next_extras['episode_success'][hard_env_id])} "
                            f"timeout={bool(next_extras['episode_timeout'][hard_env_id])} "
                            f"terminal_force={next_extras.get('log', {}).get('Metrics/contact_force_n')} "
                            f"trace={post_reset_trace}"
                        )
                        assert float(
                            unwrapped._cave_contact_force_n[hard_env_id]
                        ) <= float(cfg.collision_force_threshold_n), (
                            "Raw contact force stayed stale after reset: "
                            f"trace={post_reset_trace}"
                        )
                        if not torch.any(next_terminated | next_truncated):
                            assert "log" not in next_extras, (
                                "Non-terminal steps must not repeat a completed-episode log"
                            )
                    break
            assert collision_seen, (
                f"No hard-scene contact occurred within {args.collision_reset_steps} steps"
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
        if args.navigation_curriculum:
            initial_frontiers = unwrapped._exit_curriculum.distance_m.clone()
            for _ in range(cfg.navigation_curriculum_window - 1):
                unwrapped._robot.write_root_pose_to_sim(pose)
                unwrapped._robot.write_root_velocity_to_sim(
                    torch.zeros(args.num_envs, 6, device=unwrapped.device)
                )
                _, _, _, _, extras = env.step(torch.zeros(args.num_envs, 6, device=unwrapped.device))
                assert torch.all(extras["episode_success"])
                if torch.all(unwrapped._exit_curriculum.distance_m > initial_frontiers):
                    break
            torch.testing.assert_close(
                unwrapped._exit_curriculum.distance_m,
                torch.minimum(initial_frontiers * cfg.navigation_curriculum_growth, unwrapped._cave_scene_route_lengths),
            )
            _, _, terminated, truncated, _ = env.step(torch.zeros(args.num_envs, 6, device=unwrapped.device))
            assert not torch.any(terminated | truncated), "Promoted curriculum spawn must be safe"

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
        # Kit's fast shutdown otherwise exits with 0 even on an assertion.
        import omni.kit.app

        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        simulation_app.close()
