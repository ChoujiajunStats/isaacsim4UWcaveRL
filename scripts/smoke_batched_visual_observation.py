#!/usr/bin/env python3
"""Require exact legacy-packet/batched visual observations on real sensors."""

import argparse
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=48)
parser.add_argument("--output", type=Path, default=Path("outputs/batched_visual_equivalence.json"))
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


def main():
    cfg = parse_env_cfg("Isaac-Underwater-Cave-Navigation-v0", device=args.device, num_envs=6)
    cfg.seed = 42
    cfg.domain_randomization_enabled = False
    cfg.navigation_curriculum_enabled = False
    cfg.episode_length_s = 2.
    env = gym.make("Isaac-Underwater-Cave-Navigation-v0", cfg=cfg)
    try:
        task = env.unwrapped
        env.reset()
        resets = 0
        with torch.inference_mode():
            for step in range(args.steps + 1):
                task.cfg.batched_visual_observation_enabled = False
                legacy = {key: value.clone() for key, value in task._get_observations().items()}
                task.cfg.batched_visual_observation_enabled = True
                batched = task._get_observations()
                for key in legacy:
                    torch.testing.assert_close(batched[key], legacy[key], rtol=0, atol=0)
                    assert torch.isfinite(batched[key]).all()
                if step < args.steps:
                    _, _, terminated, truncated, _ = env.step(torch.zeros(task.num_envs, 6, device=task.device))
                    resets += int((terminated | truncated).sum())
        if args.steps >= 40:
            assert resets >= task.num_envs, "Must test frames following an automatic reset"
        report = {"policy_observation_dim": 1553, "critic_observation_dim": 19,
                  "num_envs": task.num_envs, "compared_frames": args.steps + 1,
                  "auto_resets": resets, "rtol": 0, "atol": 0, "equivalent": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"batched_visual_equivalence: PASS {report}", flush=True)
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
