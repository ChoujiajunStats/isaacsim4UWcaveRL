#!/usr/bin/env python3
"""Headless finite-state smoke test for the underwater PointNav task."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--episode_log_path", type=str, default=None)
parser.add_argument("--current_mode", choices=("constant", "sinusoidal", "random_walk"), default=None)
parser.add_argument("--domain_randomization", action="store_true")
parser.add_argument("--hydrodynamics_preset", choices=("fast_rl", "hydro_rl", "reference"), default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaac_underwater.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    cfg = parse_env_cfg(
        "Isaac-Underwater-PointNav-Direct-v0",
        device=args.device,
        num_envs=args.num_envs,
    )
    if args.episode_log_path is not None:
        cfg.episode_log_path = args.episode_log_path
    if args.current_mode is not None:
        cfg.current_mode = args.current_mode
        cfg.current_amplitude_w_mps = (0.15, 0.05, 0.0)
        cfg.current_period_s = 10.0
    if args.hydrodynamics_preset is not None:
        cfg.hydrodynamics_preset = args.hydrodynamics_preset
    cfg.domain_randomization_enabled = args.domain_randomization
    env = gym.make("Isaac-Underwater-PointNav-Direct-v0", cfg=cfg)
    obs, _ = env.reset()
    initial_position = env.unwrapped._robot.data.root_pos_w.clone()
    for _ in range(args.steps):
        actions = 0.35 * torch.randn(args.num_envs, 4, device=env.unwrapped.device)
        obs, reward, terminated, truncated, _ = env.step(actions)
        assert torch.isfinite(obs["policy"]).all()
        assert torch.isfinite(reward).all()
        assert not torch.any(terminated & truncated)
    displacement = torch.linalg.vector_norm(env.unwrapped._robot.data.root_pos_w - initial_position, dim=-1).mean()
    assert displacement > 0.01
    print(f"envs={args.num_envs} steps={args.steps} mean_displacement_m={displacement.item():.4f}")
    print("pointnav_smoke: PASS")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
