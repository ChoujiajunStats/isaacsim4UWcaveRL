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
parser.add_argument("--max_rollout_steps", type=int, default=20000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=None, help="Optional JSON metrics output path.")
parser.add_argument(
    "--current_mode",
    choices=("none", "constant", "random_constant", "slow_varying", "sinusoidal", "random_walk"),
    default=None,
)
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--hydrodynamics_preset", choices=("fast_rl", "hydro_rl", "reference"), default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
# Visual actor checkpoints require tiled camera buffers during evaluation.
if "VisualPilot" in args.task or "Cave-Explore" in args.task or "Cave-Entry" in args.task:
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
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num_envs must be positive")
    if args.max_rollout_steps <= 0:
        raise ValueError("--max_rollout_steps must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    if args.current_mode is not None:
        env_cfg.current_mode = args.current_mode
        if args.current_mode in {"slow_varying", "sinusoidal", "random_walk"}:
            env_cfg.current_amplitude_w_mps = (0.15, 0.05, 0.0)
            env_cfg.current_period_s = 10.0
    if args.domain_randomization:
        env_cfg.domain_randomization_enabled = True
    if args.hydrodynamics_preset is not None:
        env_cfg.hydrodynamics_preset = args.hydrodynamics_preset

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
        if coverage_buffer is None or unique_voxels_buffer is None or path_length_buffer is None:
            raise RuntimeError("Task did not expose episode exploration metric extras")
        coverages = coverage_buffer[done_ids].detach().cpu().tolist()
        unique_voxels = unique_voxels_buffer[done_ids].detach().cpu().tolist()
        path_lengths = path_length_buffer[done_ids].detach().cpu().tolist()
        returns = episode_returns[done_ids].detach().cpu().tolist()
        lengths = episode_lengths[done_ids].detach().cpu().tolist()
        remaining = args.episodes - len(completed_returns)
        completed_returns.extend(float(value) for value in returns[:remaining])
        completed_lengths.extend(int(value) for value in lengths[:remaining])
        completed_success.extend(bool(value) for value in success[:remaining])
        completed_bounds.extend(bool(value) for value in bounds[:remaining])
        completed_timeouts.extend(bool(value) for value in timeouts[:remaining])
        completed_collisions.extend(bool(value) for value in collisions[:remaining])
        completed_workspace_coverage.extend(float(value) for value in coverages[:remaining])
        completed_unique_voxels.extend(int(value) for value in unique_voxels[:remaining])
        completed_path_lengths.extend(float(value) for value in path_lengths[:remaining])
        episode_returns[done_ids] = 0.0
        episode_lengths[done_ids] = 0
        if hasattr(policy, "reset"):
            policy.reset(dones)
        if len(completed_returns) >= args.episodes:
            break

    if len(completed_returns) < args.episodes:
        raise RuntimeError(
            f"Only completed {len(completed_returns)} / {args.episodes} episodes "
            f"within {args.max_rollout_steps} rollout steps"
        )

    count = len(completed_returns)
    metrics: dict[str, object] = {
        "task": args.task,
        "checkpoint": str(args.checkpoint.resolve()),
        "num_envs": args.num_envs,
        "episodes": count,
        "rollout_steps": rollout_step,
        "seed": args.seed,
        "hydrodynamics_preset": getattr(env_cfg, "hydrodynamics_preset", None),
        "domain_randomization": bool(getattr(env_cfg, "domain_randomization_enabled", False)),
        "current_mode": getattr(env_cfg, "current_mode", None),
        "mean_return": sum(completed_returns) / count,
        "mean_episode_length_steps": sum(completed_lengths) / count,
        "success_rate": sum(completed_success) / count,
        "out_of_bounds_rate": sum(completed_bounds) / count,
        "timeout_rate": sum(completed_timeouts) / count,
        "collision_rate": sum(completed_collisions) / count,
        "min_return": min(completed_returns),
        "max_return": max(completed_returns),
    }
    if bool(getattr(env_cfg, "exploration_reward_enabled", False)):
        metrics.update(
            {
                "mean_workspace_coverage_fraction": sum(completed_workspace_coverage) / count,
                "mean_unique_voxels": sum(completed_unique_voxels) / count,
                "mean_path_length_m": sum(completed_path_lengths) / count,
                "coverage_denominator": "bounded_workspace_voxels_not_inferred_free_space",
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
