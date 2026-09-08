#!/usr/bin/env python3
"""Run finite PPO rollouts and write episode-level evaluation metrics.

This deliberately uses the same task registration, environment configuration,
RSL-RL wrapper, and checkpoint loader as training.  It is intended for quick
regression checks and for comparing checkpoints without entering the infinite
Isaac Lab player loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint for finite episodes.")
parser.add_argument("--task", type=str, default="Isaac-Underwater-PointNav-Direct-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument(
    "--episodes_per_scene",
    type=int,
    default=None,
    help="For multi-cave tasks, collect this many episodes from every selected scene.",
)
parser.add_argument("--max_rollout_steps", type=int, default=20000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Optional finite-evaluation horizon override; omit for the task horizon.",
)
parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics output path.")
parser.add_argument(
    "--current_mode",
    choices=("none", "constant", "random_constant", "slow_varying", "sinusoidal", "random_walk"),
    default=None,
)
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument(
    "--cave_dataset_profile",
    default=None,
    help="Override the multi-cave manifest profile before task registration.",
)
parser.add_argument("--hydrodynamics_preset", choices=("fast_rl", "hydro_rl", "reference"), default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.cave_dataset_profile:
    import os

    os.environ["ISAAC_UNDERWATER_CAVE_PROFILE"] = args.cave_dataset_profile
# Visual actor checkpoints require tiled camera buffers during evaluation.
if any(
    marker in args.task
    for marker in ("VisualPilot", "Cave-Explore", "Cave-Entry", "Cave-Navigation")
):
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import isaac_underwater.tasks  # noqa: F401
from isaac_underwater.learning import register_rsl_rl_extensions
from isaac_underwater.tasks.direct.agents import make_runner_cfg
from isaaclab.envs import DirectRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg


def _write_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if args.episodes_per_scene is None and args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episodes_per_scene is not None and args.episodes_per_scene <= 0:
        raise ValueError("--episodes_per_scene must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.max_rollout_steps <= 0:
        raise ValueError("--max_rollout_steps must be positive")
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        raise ValueError("--episode_length_s must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s
    if args.current_mode is not None:
        env_cfg.current_mode = args.current_mode
        if args.current_mode in {"slow_varying", "sinusoidal", "random_walk"}:
            env_cfg.current_amplitude_w_mps = (0.15, 0.05, 0.0)
            env_cfg.current_period_s = 10.0
    if args.domain_randomization:
        env_cfg.domain_randomization_enabled = True
    if args.hydrodynamics_preset is not None:
        env_cfg.hydrodynamics_preset = args.hydrodynamics_preset
    variants = tuple(getattr(env_cfg.world, "scene_variants", ()))
    if args.episodes_per_scene is not None and not variants:
        raise ValueError("--episodes_per_scene requires a multi-cave task profile")
    required_episodes = (
        args.episodes_per_scene * len(variants)
        if args.episodes_per_scene is not None
        else args.episodes
    )
    scene_episode_counts = {scene_id: 0 for scene_id in range(len(variants))}

    print(f"ppo_eval: building {args.task} with {args.num_envs} environments", flush=True)
    env = gym.make(args.task, cfg=env_cfg)
    if not isinstance(env.unwrapped, DirectRLEnv):
        raise TypeError(f"Expected DirectRLEnv, got {type(env.unwrapped)}")

    register_rsl_rl_extensions()
    agent_cfg = make_runner_cfg(args.task)
    agent_cfg.device = args.device or agent_cfg.device
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=vec_env.device)

    episode_returns = torch.zeros(vec_env.num_envs, device=vec_env.device)
    episode_lengths = torch.zeros(vec_env.num_envs, dtype=torch.long, device=vec_env.device)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    completed_success: list[bool] = []
    completed_bounds: list[bool] = []
    completed_timeouts: list[bool] = []
    completed_collisions: list[bool] = []
    completed_workspace_coverage: list[float] = []
    completed_unique_voxels: list[int] = []
    completed_path_lengths: list[float] = []
    completed_reference_paths: list[float] = []
    completed_spl: list[float] = []
    completed_scene_ids: list[int] = []

    obs = vec_env.get_observations()
    for rollout_step in range(1, args.max_rollout_steps + 1):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = vec_env.step(actions)
        if not torch.isfinite(actions).all() or not torch.isfinite(rewards).all():
            raise FloatingPointError(f"Non-finite policy output at rollout step {rollout_step}")
        episode_returns += rewards
        episode_lengths += 1
        done_ids = dones.to(dtype=torch.bool).nonzero(as_tuple=False).flatten()
        if done_ids.numel() == 0:
            continue

        # DirectRLEnv resets finished environments before returning from step.
        # The task preserves terminal causes in explicit extras so evaluation
        # does not accidentally read the freshly-reset flags.
        success_buffer = extras.get("episode_success")
        bounds_buffer = extras.get("episode_out_of_bounds")
        timeout_buffer = extras.get("episode_timeout")
        collision_buffer = extras.get("episode_collision")
        if success_buffer is None or bounds_buffer is None or timeout_buffer is None or collision_buffer is None:
            raise RuntimeError("Task did not expose terminal episode outcome extras")
        success = success_buffer[done_ids].detach().cpu().tolist()
        bounds = bounds_buffer[done_ids].detach().cpu().tolist()
        timeouts = timeout_buffer[done_ids].detach().cpu().tolist()
        collisions = collision_buffer[done_ids].detach().cpu().tolist()
        coverage_buffer = extras.get("episode_workspace_coverage")
        unique_voxels_buffer = extras.get("episode_unique_voxels")
        path_length_buffer = extras.get("episode_path_length_m")
        reference_path_buffer = extras.get("episode_reference_path_m")
        spl_buffer = extras.get("episode_spl")
        scene_id_buffer = extras.get("episode_scene_id")
        if any(
            value is None
            for value in (
                coverage_buffer,
                unique_voxels_buffer,
                path_length_buffer,
                reference_path_buffer,
                spl_buffer,
                scene_id_buffer,
            )
        ):
            raise RuntimeError("Task did not expose complete episode metric extras")
        coverages = coverage_buffer[done_ids].detach().cpu().tolist()
        unique_voxels = unique_voxels_buffer[done_ids].detach().cpu().tolist()
        path_lengths = path_length_buffer[done_ids].detach().cpu().tolist()
        reference_paths = reference_path_buffer[done_ids].detach().cpu().tolist()
        spl_values = spl_buffer[done_ids].detach().cpu().tolist()
        scene_ids = scene_id_buffer[done_ids].detach().cpu().tolist()
        returns = episode_returns[done_ids].detach().cpu().tolist()
        lengths = episode_lengths[done_ids].detach().cpu().tolist()
        for index in range(len(returns)):
            scene_id = int(scene_ids[index])
            if args.episodes_per_scene is not None:
                if scene_id not in scene_episode_counts:
                    raise RuntimeError(f"Task reported invalid cave scene id {scene_id}")
                if scene_episode_counts[scene_id] >= args.episodes_per_scene:
                    continue
            elif len(completed_returns) >= required_episodes:
                break
            completed_returns.append(float(returns[index]))
            completed_lengths.append(int(lengths[index]))
            completed_success.append(bool(success[index]))
            completed_bounds.append(bool(bounds[index]))
            completed_timeouts.append(bool(timeouts[index]))
            completed_collisions.append(bool(collisions[index]))
            completed_workspace_coverage.append(float(coverages[index]))
            completed_unique_voxels.append(int(unique_voxels[index]))
            completed_path_lengths.append(float(path_lengths[index]))
            completed_reference_paths.append(float(reference_paths[index]))
            completed_spl.append(float(spl_values[index]))
            completed_scene_ids.append(scene_id)
            if args.episodes_per_scene is not None:
                scene_episode_counts[scene_id] += 1
        episode_returns[done_ids] = 0.0
        episode_lengths[done_ids] = 0
        if hasattr(policy, "reset"):
            policy.reset(dones)
        if len(completed_returns) >= required_episodes:
            break

    if len(completed_returns) < required_episodes:
        raise RuntimeError(
            f"Only completed {len(completed_returns)} / {required_episodes} episodes "
            f"within {args.max_rollout_steps} rollout steps"
        )

    count = len(completed_returns)
    metrics: dict[str, object] = {
        "task": args.task,
        "checkpoint": str(args.checkpoint.resolve()),
        "num_envs": args.num_envs,
        "episodes": count,
        "episodes_per_scene": args.episodes_per_scene,
        "rollout_steps": rollout_step,
        "seed": args.seed,
        "episode_length_s": float(env_cfg.episode_length_s),
        "hydrodynamics_preset": getattr(env_cfg, "hydrodynamics_preset", None),
        "domain_randomization": bool(getattr(env_cfg, "domain_randomization_enabled", False)),
        "current_mode": getattr(env_cfg, "current_mode", None),
        "mean_return": sum(completed_returns) / count,
        "mean_episode_length_steps": sum(completed_lengths) / count,
        "success_rate": sum(completed_success) / count,
        "out_of_bounds_rate": sum(completed_bounds) / count,
        "timeout_rate": sum(completed_timeouts) / count,
        "collision_rate": sum(completed_collisions) / count,
        "mean_path_length_m": sum(completed_path_lengths) / count,
        "mean_reference_path_length_m": sum(completed_reference_paths) / count,
        "mean_spl": sum(completed_spl) / count,
        "spl_reference": "provided_collision_checked_A_star_route_not_exact_continuous_geodesic",
        "min_return": min(completed_returns),
        "max_return": max(completed_returns),
    }
    if bool(getattr(env_cfg, "exploration_reward_enabled", False)):
        metrics.update(
            {
                "mean_workspace_coverage_fraction": sum(completed_workspace_coverage) / count,
                "mean_unique_voxels": sum(completed_unique_voxels) / count,
                "coverage_denominator": "bounded_workspace_voxels_not_inferred_free_space",
            }
        )
    if variants:
        per_scene: dict[str, object] = {}
        per_difficulty_samples: dict[str, list[int]] = {}
        for scene_id, scene in enumerate(variants):
            sample_ids = [index for index, value in enumerate(completed_scene_ids) if value == scene_id]
            if not sample_ids:
                continue
            per_difficulty_samples.setdefault(scene.difficulty, []).extend(sample_ids)
            scene_count = len(sample_ids)
            per_scene[scene.key] = {
                "difficulty": scene.difficulty,
                "episodes": scene_count,
                "success_rate": sum(completed_success[index] for index in sample_ids) / scene_count,
                "collision_rate": sum(completed_collisions[index] for index in sample_ids) / scene_count,
                "mean_spl": sum(completed_spl[index] for index in sample_ids) / scene_count,
                "mean_path_length_m": sum(completed_path_lengths[index] for index in sample_ids) / scene_count,
            }
        per_difficulty = {}
        for difficulty, sample_ids in per_difficulty_samples.items():
            difficulty_count = len(sample_ids)
            per_difficulty[difficulty] = {
                "episodes": difficulty_count,
                "success_rate": sum(completed_success[index] for index in sample_ids) / difficulty_count,
                "collision_rate": sum(completed_collisions[index] for index in sample_ids) / difficulty_count,
                "mean_spl": sum(completed_spl[index] for index in sample_ids) / difficulty_count,
            }
        metrics.update(
            {
                "cave_dataset": getattr(env_cfg.world, "dataset_name", None),
                "cave_dataset_profile": getattr(env_cfg.world, "dataset_profile", None),
                "scene_assignment": "static_balanced_round_robin",
                "per_scene": per_scene,
                "per_difficulty": per_difficulty,
            }
        )
    if bool(getattr(env_cfg, "cave_entry_enabled", False)):
        portals = env.unwrapped._cave_portals
        metrics.update(
            {
                "entry_definition": "automatic_skeleton_clearance_transition",
                "inferred_portal_count": len(portals),
                "portal_chainages_m": [portal.chainage_m for portal in portals],
                "portal_exterior_runs_m": [portal.exterior_run_m for portal in portals],
                "portal_local_clearances_m": [portal.local_clearance_m for portal in portals],
            }
        )
    if args.output is not None:
        _write_metrics(args.output, metrics)
    print(json.dumps(metrics, sort_keys=True), flush=True)
    print("ppo_eval: PASS", flush=True)
    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
