#!/usr/bin/env python3
"""Frozen-checkpoint action/start ablation. Diagnostic, not acceptance.

One episode per worker per case, paired seeded poses, no learning, no replay
loading, fixed curriculum frontier, and pre-autoreset terminal snapshots.
Neither reference route nor privileged state is passed to the actor.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--algorithm", choices=("ppo", "flash"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=18)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--short_distance", type=float, default=8.0)
parser.add_argument("--scopes", nargs="+", choices=("short", "frontier", "full"), default=["short", "frontier", "full"])
parser.add_argument("--modes", nargs="+", choices=("deterministic", "stochastic", "bounded_mean"),
                    default=["deterministic", "stochastic", "bounded_mean"])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if (args.output.exists() or args.num_envs <= 0 or args.num_envs % 3 or args.short_distance <= 0
        or not args.checkpoint.exists()):
    parser.error("Require a checkpoint, fresh output, positive short distance and workers divisible by three")
args.headless = args.enable_cameras = True
os.environ["ISAAC_UNDERWATER_CAVE_PROFILE"] = "train_all"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from rsl_rl.runners import OnPolicyRunner

import isaac_underwater.tasks  # noqa: F401
from isaac_underwater.learning import register_rsl_rl_extensions
from isaac_underwater.learning.bounded_actions import clipped_normal_mean, tanh_normal_mean
from isaac_underwater.learning.flash_sac import FLASHSAC_COMMIT, load_flashsac_config, make_flashsac_agent
from isaac_underwater.tasks.direct.agents import make_runner_cfg
from isaac_underwater.tasks.direct.pointnav_env import UnderwaterPointNavEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Underwater-Cave-Navigation-v0"


class DiagnosticEnvironment(UnderwaterPointNavEnv):
    def snapshot(self, i):
        return {
            "step": int(self.episode_length_buf[i]),
            "position_m": (self._robot.data.root_pos_w[i] - self.scene.env_origins[i]).tolist(),
            "orientation_wxyz": self._robot.data.root_quat_w[i].tolist(),
            "goal_distance_m": float(torch.linalg.vector_norm(self._goal_pos_w[i] - self._robot.data.root_pos_w[i])),
            "route_deviation_m": float(self._cave_centerline_distance[i]),
            "route_chainage_m": float(self._cave_route_chainage[i]),
            "contact_force_n": float(self._cave_contact_force_n[i]),
            "action": self._actions[i].tolist(),
        }

    def step(self, actions):
        self.terminal_snapshots = {}
        self._diagnostic_step = True
        try:
            return super().step(actions)
        finally:
            self._diagnostic_step = False

    def _reset_idx(self, env_ids):
        if getattr(self, "_diagnostic_step", False):
            self.terminal_snapshots = {int(i): self.snapshot(int(i)) for i in env_ids}
        super()._reset_idx(env_ids)


def write_report(report):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)


def main():
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed, cfg.episode_length_s = args.seed, 300.0
    cfg.navigation_curriculum_enabled = True
    cfg.domain_randomization_enabled = False
    env = DiagnosticEnvironment(cfg=cfg)
    scenes = [scene.key for scene in cfg.world.scene_variants]
    report = {"algorithm": args.algorithm, "checkpoint": str(args.checkpoint.resolve()),
              "seed": args.seed, "num_envs": args.num_envs, "complete": False,
              "note": "Diagnostic only; alternative inference modes are not standard acceptance",
              "short_distance_m": args.short_distance, "cases": {}}
    try:
        if args.algorithm == "ppo":
            register_rsl_rl_extensions()
            runner_cfg = make_runner_cfg(TASK)
            runner_cfg.device = args.device
            vec = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
            runner = OnPolicyRunner(vec, runner_cfg.to_dict(), log_dir=None, device=args.device)
            info = runner.load(str(args.checkpoint))["navigation_training"]
            if info["scene_keys"] != scenes:
                raise ValueError("Checkpoint scene order mismatch")
            curriculum_state = info["exit_curriculum"]
            policy = runner.alg.policy
            policy.eval()
            model = policy
        else:
            info = json.loads((args.checkpoint / "metadata.json").read_text())
            if info["scene_keys"] != scenes or info["upstream_commit"] != FLASHSAC_COMMIT:
                raise ValueError("Checkpoint scene/revision mismatch")
            overrides = info["config"].copy()
            for key in ("device_type", "buffer_device_type"):
                overrides.pop(key)
            # Inference-only: no saved replay or optimizer loaded/modified.
            overrides.update(buffer_min_length=args.num_envs, buffer_max_length=args.num_envs,
                             sample_batch_size=2, load_optimizer=False)
            agent_cfg = load_flashsac_config(device=args.device, **overrides)
            agent = make_flashsac_agent(info["actor_observation_dim"],
                                       info["combined_critic_observation_dim"] - info["actor_observation_dim"],
                                       env.num_envs, agent_cfg)
            agent.load(str(args.checkpoint))
            curriculum_state = info["exit_curriculum"]
            model = agent._actor.network
            torch.backends.cuda.matmul.allow_tf32 = bool(info.get("tf32", False))
            torch.backends.cudnn.allow_tf32 = bool(info.get("tf32", False))
        frozen_weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        curriculum = env._exit_curriculum
        # A tuned checkpoint can have slower promotion settings. Diagnostics
        # freeze these settings and never use its randomized rehearsal sampler.
        for key in ("window", "success_threshold", "growth"):
            setattr(curriculum, key, curriculum_state[key])
        curriculum.load_state_dict(curriculum_state)
        # All cases have fixed starts. Completed extra episodes may not promote them.
        curriculum.record = lambda *arguments: None
        with torch.inference_mode():
            for scope in args.scopes:
                paired_poses = None
                for mode in args.modes:
                    state = copy.deepcopy(curriculum_state)
                    if scope == "short":
                        state["distance_m"] = [min(args.short_distance, length) for length in state["route_lengths_m"]]
                        state["outcomes"] = [[] for _ in scenes]
                    curriculum.load_state_dict(state)
                    env._exit_curriculum = None if scope == "full" else curriculum
                    env.seed(args.seed)
                    obs, _ = env.reset()
                    if args.algorithm == "ppo":
                        policy.reset()
                    else:
                        agent._cur_noise_repeat_count.zero_()
                    initial = [env.snapshot(i) for i in range(env.num_envs)]
                    poses = torch.cat((env._robot.data.root_pos_w, env._robot.data.root_quat_w), dim=1).clone()
                    if paired_poses is not None:
                        torch.testing.assert_close(poses, paired_poses, rtol=0, atol=0)
                    paired_poses = poses
                    complete = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
                    statistics = torch.zeros(env.num_envs, 4, device=env.device)
                    totals = torch.zeros(env.num_envs, device=env.device)
                    rows = []
                    for step in range(1, env.max_episode_length + 1):
                        if args.algorithm == "ppo":
                            # Exactly one recurrent forward/step for every inference mode.
                            sampled = policy.act(obs)
                            mean, std = policy.action_mean, policy.action_std
                            actions = (sampled if mode == "stochastic" else
                                       clipped_normal_mean(mean, std) if mode == "bounded_mean" else mean).clamp(-1, 1)
                        else:
                            mean, std = agent._actor.apply("get_mean_and_std", observations=obs["policy"], training=False)
                            if mode == "stochastic":
                                actions = torch.as_tensor(agent.sample_actions(
                                    step, {"next_observation": obs["policy"]}, training=True), device=env.device)
                            else:
                                actions = tanh_normal_mean(mean, std) if mode == "bounded_mean" else mean.tanh()
                        if not torch.isfinite(actions).all():
                            raise FloatingPointError("Nonfinite action")
                        active = ~complete
                        statistics[:, 0] += active
                        statistics[:, 1] += active * std[:, :4].mean(-1)
                        statistics[:, 2] += active * (actions[:, :4].abs() >= .98).float().mean(-1)
                        statistics[:, 3] += active * actions[:, :4].abs().mean(-1)
                        obs, reward, terminated, truncated, extras = env.step(actions)
                        totals += active * reward
                        dones = terminated | truncated
                        for i in (dones & active).nonzero().flatten().tolist():
                            scene_id = int(extras["episode_scene_id"][i])
                            values = statistics[i].tolist()
                            row = {"env_id": i, "scene": scenes[scene_id], "initial": initial[i],
                                   "terminal": env.terminal_snapshots[i], "return": float(totals[i]),
                                   "mean_raw_std_motion": values[1] / values[0],
                                   "motion_saturation_fraction": values[2] / values[0],
                                   "mean_abs_motion_action": values[3] / values[0],
                                   "initial_remaining_route_m": float(extras["episode_reference_path_m"][i]),
                                   **{name: bool(extras[f"episode_{name}"][i])
                                      for name in ("success", "collision", "out_of_bounds", "timeout")},
                                   "path_length_m": float(extras["episode_path_length_m"][i])}
                            rows.append(row)
                            complete[i] = True
                        if args.algorithm == "ppo":
                            policy.reset(dones)
                        if step % 500 == 0:
                            print(f"action_diag: {scope}/{mode} step={step} complete={len(rows)}/{env.num_envs}", flush=True)
                        if bool(complete.all()):
                            break
                    if not bool(complete.all()):
                        raise RuntimeError(f"Incomplete diagnostic {scope}/{mode}")
                    for key, original in frozen_weights.items():
                        torch.testing.assert_close(model.state_dict()[key], original, rtol=0, atol=0)
                    per_scene = {scene: {"episodes": sum(row["scene"] == scene for row in rows),
                                        **{outcome: sum(row[outcome] for row in rows if row["scene"] == scene)
                                           for outcome in ("success", "collision", "out_of_bounds", "timeout")}}
                                 for scene in scenes}
                    report["cases"][f"{scope}/{mode}"] = {"per_scene": per_scene, "episodes": rows,
                                                              "paired_initial_poses_exact": True,
                                                              "actor_weights_and_buffers_unchanged": True}
                    write_report(report)
                    print(f"action_diag: RESULT {scope}/{mode} {json.dumps(per_scene)}", flush=True)
        report["complete"] = True
        write_report(report)
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
