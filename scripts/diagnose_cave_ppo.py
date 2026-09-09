#!/usr/bin/env python3
"""Read-only PPO trajectory diagnostics, not a statistical acceptance test.

Record one seeded episode per cave at the real entrance and the saved
curriculum frontier. Capture terminal state before automatic reset, without
changing observations, actions, rewards or termination rules.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--profile", default="train_all")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sample_interval", type=int, default=10)
parser.add_argument("--output", type=Path, default=Path("outputs/cave_assets/ppo_failure_traces.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.sample_interval <= 0 or not args.checkpoint.is_file():
    parser.error("A real checkpoint and positive sample interval are required")
args.headless = True
args.enable_cameras = True
os.environ["ISAAC_UNDERWATER_CAVE_PROFILE"] = args.profile
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

import isaac_underwater.tasks  # noqa: F401
from isaac_underwater.learning import register_rsl_rl_extensions
from isaac_underwater.navigation.route_geometry import sample_polyline
from isaac_underwater.tasks.direct.agents import make_runner_cfg
from isaac_underwater.tasks.direct.pointnav_env import UnderwaterPointNavEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Underwater-Cave-Navigation-v0"


class TraceEnvironment(UnderwaterPointNavEnv):
    def __init__(self, *arguments, **kwargs):
        self._trace_active = False
        self.terminal_snapshots = {}
        super().__init__(*arguments, **kwargs)

    def snapshot(self, env_id):
        scene_id = int(self._cave_scene_ids[env_id])
        position = self._robot.data.root_pos_w[env_id] - self.scene.env_origins[env_id]
        goal_delta = self._goal_pos_w[env_id] - self._robot.data.root_pos_w[env_id]
        velocity = self._robot.data.root_lin_vel_w[env_id]
        speed = torch.linalg.vector_norm(velocity).clamp_min(1.e-6)
        distance = torch.linalg.vector_norm(goal_delta)
        _, tangent = sample_polyline(
            self._cave_route_chainage[env_id:env_id+1], self._multi_cave_centerlines[scene_id:scene_id+1],
            self._multi_cave_chainages[scene_id:scene_id+1], self._multi_cave_route_mask[scene_id:scene_id+1],
        )
        return {
            "step": int(self.episode_length_buf[env_id]), "position_m": position.tolist(),
            "velocity_world_mps": velocity.tolist(), "action": self._actions[env_id].tolist(),
            "goal_distance_m": float(distance), "route_deviation_m": float(self._cave_centerline_distance[env_id]),
            "route_chainage_m": float(self._cave_route_chainage[env_id]),
            "velocity_goal_alignment": float(torch.dot(velocity, goal_delta) / (speed * distance.clamp_min(1.e-6))),
            "velocity_route_alignment": float(torch.dot(velocity, tangent[0]) / speed),
            "contact_force_n": float(self._cave_contact_force_n[env_id]),
        }

    def step(self, actions):
        self.terminal_snapshots = {}
        self._trace_active = True
        try:
            return super().step(actions)
        finally:
            self._trace_active = False

    def _reset_idx(self, env_ids):
        if self._trace_active:
            self.terminal_snapshots = {int(i): self.snapshot(int(i)) for i in env_ids}
        super()._reset_idx(env_ids)


def main():
    cfg = parse_env_cfg(TASK, device=args.device)
    cfg.scene.num_envs = len(cfg.world.scene_variants)
    cfg.seed = args.seed
    cfg.navigation_curriculum_enabled = True
    cfg.domain_randomization_enabled = False
    env = TraceEnvironment(cfg=cfg)
    register_rsl_rl_extensions()
    runner_cfg = make_runner_cfg(TASK)
    runner_cfg.device = args.device
    vec = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    try:
        runner = OnPolicyRunner(vec, runner_cfg.to_dict(), log_dir=None, device=args.device)
        infos = runner.load(str(args.checkpoint))
        navigation_state = infos["navigation_training"]
        if navigation_state["scene_keys"] != [scene.key for scene in cfg.world.scene_variants]:
            raise ValueError("Checkpoint scene identities/order mismatch")
        curriculum = env._exit_curriculum
        curriculum.load_state_dict(navigation_state["exit_curriculum"])
        policy = runner.get_inference_policy(device=vec.device)
        report = {"checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
                  "note": "One episode per scene/phase; diagnostic only, not acceptance", "phases": {}}
        with torch.inference_mode():
            for phase in ("full_route", "curriculum_frontier"):
                env._exit_curriculum = curriculum if phase == "curriculum_frontier" else None
                env.seed(args.seed)
                obs, _ = vec.reset()
                runner.alg.policy.reset()
                complete = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                scenes = {}
                for i, scene in enumerate(cfg.world.scene_variants):
                    valid = env._multi_cave_route_mask[i]
                    scenes[scene.key] = {
                        "reference_route_m": env._multi_cave_centerlines[i][valid].tolist(),
                        "goal_m": (env._goal_pos_w[i] - env.scene.env_origins[i]).tolist(),
                        "initial_remaining_route_m": float(env._episode_route_reference_m[i]),
                        "samples": [env.snapshot(i)],
                    }
                for step in range(1, env.max_episode_length + 1):
                    actions = policy(obs)
                    obs, reward, dones, extras = vec.step(actions)
                    if not torch.isfinite(actions).all() or not torch.isfinite(reward).all():
                        raise FloatingPointError("Nonfinite policy action/reward")
                    for i, scene in enumerate(cfg.world.scene_variants):
                        if bool(complete[i]):
                            continue
                        record = scenes[scene.key]
                        if bool(dones[i]):
                            record["samples"].append(env.terminal_snapshots[i])
                            record["outcome"] = {name: bool(extras[f"episode_{name}"][i])
                                                 for name in ("success", "collision", "out_of_bounds", "timeout")}
                            record["outcome"]["path_length_m"] = float(extras["episode_path_length_m"][i])
                            complete[i] = True
                            print(f"ppo_trace: {phase} {scene.key} outcome={record['outcome']} "
                                  f"terminal={record['samples'][-1]}", flush=True)
                        elif step % args.sample_interval == 0:
                            record["samples"].append(env.snapshot(i))
                    runner.alg.policy.reset(dones)
                    if bool(torch.all(complete)):
                        break
                if not bool(torch.all(complete)):
                    raise RuntimeError(f"Incomplete diagnostic phase {phase}")
                report["phases"][phase] = scenes
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"ppo_trace: completed {args.output}", flush=True)
    finally:
        vec.close()


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
