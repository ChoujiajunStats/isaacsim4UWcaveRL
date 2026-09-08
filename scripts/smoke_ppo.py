#!/usr/bin/env python3
"""Finite headless checkpoint-load and inference smoke test for RSL-RL."""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Underwater-PointNav-Direct-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument(
    "--cave_dataset_profile",
    default=None,
    help="Override the multi-cave manifest profile before task registration.",
)
parser.add_argument("--domain_randomization", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.cave_dataset_profile:
    os.environ["ISAAC_UNDERWATER_CAVE_PROFILE"] = args.cave_dataset_profile
if args.domain_randomization:
    os.environ["ISAAC_UNDERWATER_MULTICAVE_DOMAIN_RANDOMIZATION"] = "1"
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


def main() -> None:
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    print("ppo_smoke: building environment", flush=True)
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = 42
    if args.domain_randomization:
        env_cfg.domain_randomization_enabled = True
    env = gym.make(args.task, cfg=env_cfg)
    if not isinstance(env.unwrapped, DirectRLEnv):
        raise TypeError(f"Expected DirectRLEnv, got {type(env.unwrapped)}")

    register_rsl_rl_extensions()
    agent_cfg = make_runner_cfg(args.task)
    agent_cfg.device = args.device or agent_cfg.device
    print("ppo_smoke: loading checkpoint", flush=True)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args.checkpoint))
    policy = runner.get_inference_policy(device=vec_env.device)

    print("ppo_smoke: stepping policy", flush=True)
    obs = vec_env.get_observations()
    reward_sum = torch.zeros(vec_env.num_envs, device=vec_env.device)
    for _ in range(args.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, _ = vec_env.step(actions)
        assert torch.isfinite(actions).all()
        assert torch.isfinite(rewards).all()
        reward_sum += rewards
        with torch.inference_mode():
            runner.alg.policy.reset(dones)

    assert torch.isfinite(reward_sum).all()
    print(
        f"checkpoint={args.checkpoint} envs={vec_env.num_envs} steps={args.steps} "
        f"mean_reward={reward_sum.mean().item():.4f}",
        flush=True,
    )
    print("ppo_smoke: PASS", flush=True)
    vec_env.close()


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
