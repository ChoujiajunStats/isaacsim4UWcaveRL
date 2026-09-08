#!/usr/bin/env python3
"""Check route traversability with a privileged path follower, not a learned policy."""

import argparse
import json
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", default="train_all")
parser.add_argument("--episode_length_s", type=float, default=300.0)
parser.add_argument("--lookahead_m", type=float, default=0.8)
parser.add_argument("--speed_mps", type=float, default=1.2)
parser.add_argument("--output", type=Path, default=Path("outputs/cave_assets/route_following.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
os.environ["ISAAC_UNDERWATER_CAVE_PROFILE"] = args.profile
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import isaac_underwater.tasks  # noqa: F401
from isaac_underwater.navigation.route_geometry import project_polyline, sample_polyline
from isaac_underwater.physics import rotate_world_to_body
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg("Isaac-Underwater-Cave-Navigation-v0", device=args.device)
    cfg.scene.num_envs = len(cfg.world.scene_variants)
    cfg.seed = 42
    cfg.episode_length_s = args.episode_length_s
    cfg.navigation_curriculum_enabled = False
    cfg.domain_randomization_enabled = False
    cfg.camera_sensor = cfg.camera_left_sensor = cfg.camera_right_sensor = cfg.imu_sensor = None
    cfg.visual_observation_enabled = False
    cfg.observation_space = 19
    cfg.state_space = 0
    cfg.world.visual_enabled = False
    cfg.detailed_robot_visual = False
    cfg.lighting.enabled = False
    env = gym.make("Isaac-Underwater-Cave-Navigation-v0", cfg=cfg)
    try:
        task = env.unwrapped
        env.reset()
        scene_ids = task._cave_scene_ids
        points = task._multi_cave_centerlines[scene_ids]
        chainages = task._multi_cave_chainages[scene_ids]
        valid = task._multi_cave_route_mask[scene_ids]
        goal_chainages = task._cave_scene_goal_chainages[scene_ids]
        results = {}
        max_deviation = torch.zeros(task.num_envs, device=task.device)
        completed = torch.zeros(task.num_envs, dtype=torch.bool, device=task.device)
        with torch.inference_mode():
            for step in range(1, task.max_episode_length + 1):
                position = task._robot.data.root_pos_w - task.scene.env_origins
                distance, chainage, _ = project_polyline(position, points, chainages, valid)
                max_deviation = torch.where(completed, max_deviation, torch.maximum(max_deviation, distance))
                target, tangent = sample_polyline(
                    torch.minimum(chainage + args.lookahead_m, goal_chainages), points, chainages, valid
                )
                delta_b = rotate_world_to_body(task._robot.data.root_quat_w, target - position)
                direction_b = rotate_world_to_body(task._robot.data.root_quat_w, tangent)
                velocity_b = 2.0 * delta_b
                speed = torch.linalg.vector_norm(velocity_b, dim=-1, keepdim=True)
                velocity_b *= (args.speed_mps / speed.clamp_min(1.e-6)).clamp(max=1.0)
                actions = torch.zeros(task.num_envs, 6, device=task.device)
                actions[:, :3] = velocity_b / task._max_command_velocity
                actions[:, 3] = 2.0 * torch.atan2(direction_b[:, 1], direction_b[:, 0]) / cfg.max_command_yaw_rate_radps
                actions[completed] = 0.0
                _, _, terminated, truncated, extras = env.step(actions.clamp(-1.0, 1.0))
                newly_done = (terminated | truncated) & ~completed
                for env_id in newly_done.nonzero().flatten().tolist():
                    key = cfg.world.scene_variants[env_id].key
                    results[key] = {
                        "success": bool(extras["episode_success"][env_id]),
                        "collision": bool(extras["episode_collision"][env_id]),
                        "out_of_bounds": bool(extras["episode_out_of_bounds"][env_id]),
                        "timeout": bool(extras["episode_timeout"][env_id]),
                        "elapsed_s": round(step * task.step_dt, 3),
                        "path_length_m": float(extras["episode_path_length_m"][env_id]),
                        "max_route_deviation_m": float(max_deviation[env_id]),
                    }
                    print(f"route_oracle: {key} {results[key]}", flush=True)
                completed |= newly_done
                if torch.all(completed):
                    break
        report = {
            "controller": "privileged_reference_path_follower_not_PPO",
            "profile": args.profile,
            "episode_length_s": args.episode_length_s,
            "lookahead_m": args.lookahead_m,
            "command_speed_mps": args.speed_mps,
            "per_scene": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert len(results) == task.num_envs and all(
            result["success"] and not result["collision"] for result in results.values()
        ), f"Reference follower failed: {report}"
        print("cave_route_following: PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        import omni.kit.app

        omni.kit.app.get_app().post_quit(1)
        raise
    finally:
        simulation_app.close()
