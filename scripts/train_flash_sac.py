#!/usr/bin/env python3
"""Finite official FlashSAC pilot on the existing visual multi-cave task.

Native feed-forward FlashSAC visual MLP; not a matched CNN+GRU PPO comparison.
No upstream Isaac wrapper: cameras stay enabled and replay gets pre-reset
terminal observations. Nominal sensors / ground-truth localization only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--train_steps", type=int, default=4096, help="Additional vector steps, NOT PPO iterations.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="native_ff_pilot")
parser.add_argument("--cave_dataset_profile", default="train_all")
parser.add_argument("--navigation_curriculum", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--warmup_transitions", type=int, default=None)
parser.add_argument("--replay_capacity", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--save_interval_steps", type=int, default=1024)
parser.add_argument("--save_replay", action="store_true", help="Save replay once with the final checkpoint.")
parser.add_argument("--save_replay_checkpoints", action="store_true", help="Also save replay at periodic checkpoints for long-job recovery.")
parser.add_argument("--checkpoint", type=Path, help="Our FlashSAC directory, not a PPO .pt file; 0 train steps = evaluate.")
parser.add_argument("--resize_resume", action="store_true", help="Explicitly allow changing worker count, growing replay and changing batch size on resume.")
parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None,
                    help="Opt into upstream's faster TF32 matmul; loading otherwise preserves checkpoint mode.")
parser.add_argument("--eval_episodes_per_scene", type=int, default=3, help="0 skips full-route evaluation.")
parser.add_argument("--max_eval_steps", type=int, default=20000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.train_steps < 0 or args.save_interval_steps <= 0:
    parser.error("num_envs/save_interval_steps must be positive; train_steps must be nonnegative")
if args.eval_episodes_per_scene < 0 or args.max_eval_steps <= 0:
    parser.error("Invalid finite evaluation budget")
if args.train_steps == 0 and (args.checkpoint is None or args.eval_episodes_per_scene == 0):
    parser.error("Zero training steps requires a FlashSAC checkpoint and evaluation episodes")
if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
    parser.error("run_name must be a single directory-name component")
args.headless = True
args.enable_cameras = True
os.environ["ISAAC_UNDERWATER_CAVE_PROFILE"] = args.cave_dataset_profile
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import isaac_underwater.tasks  # noqa: F401
from isaac_underwater.learning.flash_sac import (
    FLASHSAC_COMMIT, PROJECT_ROOT, TerminalObservationMixin, load_flashsac_config,
    make_flashsac_agent, pack_flashsac_observation, replay_storage_bytes,
    restore_flashsac_training_state,
)
from isaac_underwater.navigation.evaluation_quota import SceneEpisodeQuota
from isaac_underwater.tasks.direct.pointnav_env import UnderwaterPointNavEnv
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Underwater-Cave-Navigation-v0"
ARCHITECTURE = "native_feedforward_flashsac_visual_mlp"


class FlashCaveEnvironment(TerminalObservationMixin, UnderwaterPointNavEnv):
    pass


def write_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def episode_row(env_id, extras, returns, lengths):
    row = {"environment_id": env_id, "return": float(returns[env_id]), "length_steps": int(lengths[env_id])}
    for name, key, convert in (
        ("scene_id", "episode_scene_id", int),
        ("success", "episode_success", bool),
        ("collision", "episode_collision", bool),
        ("out_of_bounds", "episode_out_of_bounds", bool),
        ("timeout", "episode_timeout", bool),
        ("path_length_m", "episode_path_length_m", float),
        ("reference_path_length_m", "episode_reference_path_m", float),
        ("spl", "episode_spl", float),
    ):
        row[name] = convert(extras[key][env_id])
    if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
        raise FloatingPointError(f"Nonfinite episode metrics: {row}")
    return row


def summarize(rows):
    result = {"episodes": len(rows)}
    if not rows:
        return result
    for key, target in (
        ("return", "mean_return"), ("length_steps", "mean_episode_length_steps"),
        ("success", "success_rate"), ("collision", "collision_rate"),
        ("out_of_bounds", "out_of_bounds_rate"), ("timeout", "timeout_rate"),
        ("path_length_m", "mean_path_length_m"), ("reference_path_length_m", "mean_reference_path_length_m"),
        ("spl", "mean_spl"),
    ):
        result[target] = sum(row[key] for row in rows) / len(rows)
    return result


def save_checkpoint(agent, env, cfg, path, vector_steps, transitions, save_replay):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    agent.save(str(path))
    if save_replay:
        agent.save_replay_buffer(str(path))
    metadata = {
        "version": 1, "upstream_commit": FLASHSAC_COMMIT, "architecture": ARCHITECTURE,
        "config": asdict(cfg), "task": TASK, "num_envs": env.num_envs,
        "actor_observation_dim": agent._actor_observation_dim,
        "combined_critic_observation_dim": agent._critic_observation_dim,
        "scene_keys": [scene.key for scene in env.cfg.world.scene_variants],
        "cave_dataset_profile": args.cave_dataset_profile,
        "vector_steps": vector_steps, "transitions": transitions,
        "resumed_from": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "resized_resume": args.resize_resume,
        "tf32": args.tf32,
        "gradient_updates": agent._update_step, "replay_saved": save_replay,
        "replay_size": len(agent._replay_buffer),
        "navigation_curriculum": env._exit_curriculum is not None,
        "exit_curriculum": env._exit_curriculum.state_dict() if env._exit_curriculum is not None else None,
        "resume_semantics": "optimizer_and_replay_and_curriculum_resume_with_new_simulator_episodes_not_bit_exact",
    }
    write_json(path / "metadata.json", metadata)


def evaluate(agent, env, checkpoint, run_dir):
    # Real entrance, full 300-second horizon, with no curriculum frontier.
    env._exit_curriculum = None
    env.cfg.navigation_curriculum_enabled = False
    env._episode_navigation_horizon_steps.fill_(env.max_episode_length)
    env.seed(args.seed)
    with torch.no_grad():
        obs, _ = env.reset()
    variants = env.cfg.world.scene_variants
    quota = SceneEpisodeQuota(env._cave_scene_ids.tolist(), len(variants), args.eval_episodes_per_scene)
    target = len(variants) * args.eval_episodes_per_scene
    returns = torch.zeros(env.num_envs, device=env.device)
    lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    rows = []
    for step in range(1, args.max_eval_steps + 1):
        actions = torch.as_tensor(agent.sample_actions(
            step, {"next_observation": pack_flashsac_observation(obs)}, training=False,
        ), device=env.device)
        with torch.no_grad():
            obs, reward, terminated, truncated, extras = env.step(actions)
        if not torch.isfinite(actions).all() or not torch.isfinite(reward).all():
            raise FloatingPointError(f"Nonfinite evaluation action/reward at {step}")
        returns += reward
        lengths += 1
        done_ids = (terminated | truncated).nonzero().flatten().tolist()
        for env_id in done_ids:
            row = episode_row(env_id, extras, returns, lengths)
            if quota.accept(env_id, row["scene_id"]):
                rows.append(row)
                print(f"flashsac_eval: {len(rows)}/{target} scene={variants[row['scene_id']].key} {row}", flush=True)
        returns[done_ids] = 0
        lengths[done_ids] = 0
        if step % 500 == 0:
            print(f"flashsac_eval: step={step} episodes={len(rows)}/{target}", flush=True)
        if len(rows) == target:
            break
    if len(rows) != target:
        write_json(run_dir / "evaluation_incomplete.json", {"episodes": rows, "rollout_steps": step, "complete": False})
        raise RuntimeError(f"Evaluation incomplete: {len(rows)}/{target} episodes in {step} steps")
    result = {
        **summarize(rows), "task": TASK, "algorithm": "FlashSAC", "architecture": ARCHITECTURE,
        "checkpoint": str(checkpoint.resolve()), "num_envs": env.num_envs, "seed": args.seed,
        "upstream_commit": FLASHSAC_COMMIT, "rollout_steps": step, "tf32": args.tf32,
        "episode_length_s": env.cfg.episode_length_s, "navigation_curriculum": False,
        "domain_randomization": False, "policy_state_source": env.cfg.policy_state_source,
        "cave_dataset_profile": args.cave_dataset_profile, "cave_dataset": env.cfg.world.dataset_name,
        "episodes_per_scene": args.eval_episodes_per_scene,
        "episode_allocation": "fixed_balanced_per_environment", "episode_quotas": quota.quotas,
        "accepted_episodes_per_environment": quota.counts,
        "spl_reference": "provided_collision_checked_A_star_route_not_exact_continuous_geodesic",
        "per_scene": {scene.key: {**summarize([row for row in rows if row["scene_id"] == i]),
                                  "difficulty": scene.difficulty} for i, scene in enumerate(variants)},
        "episode_records": rows,
    }
    write_json(run_dir / "evaluation_full.json", result)
    print(f"flashsac_eval: completed success={result['success_rate']:.3f} collision={result['collision_rate']:.3f} "
          f"SPL={result['mean_spl']:.3f}; completion is NOT navigation acceptance", flush=True)


def main():
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.episode_length_s = 300.0
    env_cfg.navigation_curriculum_enabled = args.navigation_curriculum and args.train_steps > 0
    env_cfg.domain_randomization_enabled = False
    if env_cfg.policy_state_source != "ground_truth":
        raise ValueError("Terminal observation capture is currently validated only for nominal ground-truth localization")
    if args.num_envs < len(env_cfg.world.scene_variants):
        raise ValueError("Use at least one environment per selected cave")
    cfg_overrides = {"seed": args.seed}
    for arg, key in ((args.warmup_transitions, "buffer_min_length"), (args.replay_capacity, "buffer_max_length"),
                     (args.batch_size, "sample_batch_size")):
        if arg is not None:
            cfg_overrides[key] = arg
    metadata = None
    if args.checkpoint is not None:
        metadata = json.loads((args.checkpoint / "metadata.json").read_text(encoding="utf-8"))
        if (metadata.get("version") != 1 or metadata.get("task") != TASK
                or metadata["upstream_commit"] != FLASHSAC_COMMIT or metadata["architecture"] != ARCHITECTURE):
            raise ValueError("Checkpoint algorithm/revision mismatch")
        if not args.resize_resume and any(value is not None for value in (args.warmup_transitions, args.replay_capacity, args.batch_size)):
            raise ValueError("Replay configuration comes from the checkpoint when loading")
        requested_overrides = cfg_overrides.copy()
        cfg_overrides = metadata["config"].copy()
        cfg_overrides.pop("device_type")
        cfg_overrides.pop("buffer_device_type")
        if args.resize_resume:
            if args.train_steps <= 0:
                raise ValueError("--resize_resume is for training, not evaluation")
            requested_overrides.pop("seed")
            cfg_overrides.update(requested_overrides)
            if cfg_overrides["buffer_max_length"] < metadata["config"]["buffer_max_length"]:
                raise ValueError("Resized resume cannot shrink and discard saved replay")
        if args.train_steps > 0:
            if not metadata["replay_saved"]:
                raise ValueError("Training resume requires saved replay")
            if metadata["num_envs"] != args.num_envs and not args.resize_resume:
                raise ValueError("Changing the number of environments requires --resize_resume")
            if metadata["navigation_curriculum"] != args.navigation_curriculum:
                raise ValueError("Training resume must preserve the curriculum setting")
    if args.tf32 is None:
        args.tf32 = bool(metadata.get("tf32", False)) if metadata is not None else False
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    cfg = load_flashsac_config(device=args.device, **cfg_overrides)
    run_dir = PROJECT_ROOT / "logs/flash_sac/underwater_cave_multinav" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    env = FlashCaveEnvironment(cfg=env_cfg)
    writer = SummaryWriter(str(run_dir))
    try:
        obs, _ = env.reset()
        agent = make_flashsac_agent(obs["policy"].shape[-1], obs["critic"].shape[-1], env.num_envs, cfg)
        initial_step = 0
        initial_transitions = 0
        if metadata is not None:
            if (metadata["actor_observation_dim"] != agent._actor_observation_dim
                    or metadata["combined_critic_observation_dim"] != agent._critic_observation_dim):
                raise ValueError("Checkpoint observation contract mismatch")
            if args.train_steps > 0 and metadata["scene_keys"] != [scene.key for scene in env_cfg.world.scene_variants]:
                raise ValueError("Training resume must preserve scene identities/order; evaluation may use held-out scenes")
            agent.load(str(args.checkpoint))
            if args.train_steps > 0:
                restore_flashsac_training_state(
                    agent, args.checkpoint, old_capacity=metadata["config"]["buffer_max_length"], num_envs=env.num_envs,
                )
                if env._exit_curriculum is not None:
                    env._exit_curriculum.load_state_dict(metadata["exit_curriculum"])
                initial_step = metadata["vector_steps"]
                initial_transitions = metadata["transitions"]
                obs, _ = env.reset()
        memory_bytes = replay_storage_bytes(cfg.buffer_max_length, agent._critic_observation_dim, 6)
        print(f"flashsac_train: run={run_dir} actor={agent._actor_observation_dim} "
              f"critic={agent._critic_observation_dim} replay_MiB={memory_bytes / 2**20:.3f} "
              f"vector_steps={args.train_steps} envs={env.num_envs} eager=True AMP=False TF32={args.tf32}", flush=True)
        returns = torch.zeros(env.num_envs, device=env.device)
        lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        rows = []
        terminal_observation_count = 0
        latest_losses = {}
        start = time.monotonic()
        initial_updates = agent._update_step
        for local_step in range(1, args.train_steps + 1):
            step = initial_step + local_step
            transitions = initial_transitions + local_step * env.num_envs
            packed = pack_flashsac_observation(obs)
            actions = torch.as_tensor(agent.sample_actions(step, {"next_observation": packed}, training=True), device=env.device)
            with torch.no_grad():
                obs, reward, terminated, truncated, extras = env.step(actions)
                final = pack_flashsac_observation(extras["final_observation"])
            if not all(torch.isfinite(tensor).all() for tensor in (packed, actions, reward, final)):
                raise FloatingPointError(f"Nonfinite transition at step {step}")
            # Replay uses pre-reset final; the next acting step uses autoreset obs above.
            agent.process_transition({"observation": packed, "action": actions, "reward": reward,
                                      "next_observation": final, "terminated": terminated, "truncated": truncated})
            if agent.can_start_training():
                latest_losses = agent.update()
                if not all(math.isfinite(value) for value in latest_losses.values()):
                    raise FloatingPointError(f"Nonfinite gradient-update metrics at {step}: {latest_losses}")
            returns += reward
            lengths += 1
            done_ids = (terminated | truncated).nonzero().flatten().tolist()
            if done_ids:
                if env._captured_terminal_ids is None or env._captured_terminal_ids.tolist() != done_ids:
                    raise RuntimeError("Replay terminal observation capture missed an auto-reset")
                terminal_observation_count += len(done_ids)
            for env_id in done_ids:
                rows.append(episode_row(env_id, extras, returns, lengths))
            returns[done_ids] = 0
            lengths[done_ids] = 0
            if local_step % 128 == 0 or local_step == args.train_steps:
                frontiers = env._exit_curriculum.distance_m.tolist() if env._exit_curriculum is not None else None
                recent = summarize(rows[-100:])
                elapsed = time.monotonic() - start
                print(f"flashsac_train: step={step} transitions={transitions} updates={agent._update_step} "
                      f"replay={len(agent._replay_buffer)} samples_per_s={local_step * env.num_envs / elapsed:.1f} "
                      f"frontier_m={frontiers} last100={recent} losses={latest_losses}", flush=True)
                for key, value in latest_losses.items():
                    writer.add_scalar(f"FlashSAC/{key}", value, transitions)
                for key, value in recent.items():
                    writer.add_scalar(f"CompletedEpisodes/{key}", value, transitions)
                for i, scene in enumerate(env_cfg.world.scene_variants):
                    if frontiers is not None:
                        writer.add_scalar(f"Curriculum/{scene.key}/remaining_route_m", frontiers[i], transitions)
                writer.flush()
            if local_step % args.save_interval_steps == 0 and local_step < args.train_steps:
                save_checkpoint(agent, env, cfg, run_dir / f"step_{step:07d}", step, transitions, args.save_replay_checkpoints)
        final_checkpoint = args.checkpoint
        if args.train_steps:
            final_step = initial_step + args.train_steps
            final_checkpoint = run_dir / f"step_{final_step:07d}"
            save_checkpoint(agent, env, cfg, final_checkpoint, final_step,
                            initial_transitions + args.train_steps * env.num_envs,
                            args.save_replay or args.save_replay_checkpoints)
            training_seconds = time.monotonic() - start
            # Validate native checkpoint I/O on actual camera observations before evaluating.
            probe = {"next_observation": pack_flashsac_observation(obs)}
            before = agent.sample_actions(0, probe, training=False)
            agent.load(str(final_checkpoint))
            np.testing.assert_allclose(agent.sample_actions(0, probe, training=False), before, rtol=0, atol=0)
            summary = {
                **summarize(rows), "algorithm": "FlashSAC", "architecture": ARCHITECTURE,
                "upstream_commit": FLASHSAC_COMMIT, "config": asdict(cfg), "tf32": args.tf32,
                "additional_vector_steps": args.train_steps, "additional_transitions": args.train_steps * env.num_envs,
                "additional_gradient_updates": agent._update_step - initial_updates,
                "training_seconds": training_seconds, "replay_storage_bytes": memory_bytes,
                "terminal_observations_captured": terminal_observation_count,
                "navigation_curriculum": env._exit_curriculum is not None,
                "exit_curriculum": env._exit_curriculum.state_dict() if env._exit_curriculum is not None else None,
                "checkpoint_roundtrip_actions_equal": True, "checkpoint": str(final_checkpoint),
                "per_scene": {scene.key: summarize([row for row in rows if row["scene_id"] == i])
                              for i, scene in enumerate(env_cfg.world.scene_variants)},
                "acceptance_note": "Training curriculum successes are not entrance-to-exit acceptance",
            }
            write_json(run_dir / "training_summary.json", summary)
        if args.eval_episodes_per_scene:
            evaluate(agent, env, final_checkpoint, run_dir)
        print(f"flashsac_integration: completed; results={run_dir}; navigation acceptance is separate", flush=True)
    finally:
        writer.close()
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
