#!/usr/bin/env python3
"""Durable sequential PPO + FlashSAC continuation with balanced evaluation.

Both jobs get the same additional transition budget (rounded to complete
vector rollouts), but their architectures and prior training differ. This
is not a controlled from-scratch algorithm benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from supervise_training import ROOT, run_guarded, write_status

PYTHON = str(ROOT / ".venv/bin/python")
TASK = "Isaac-Underwater-Cave-Navigation-v0"
PPO_ROOT = ROOT / "logs/rsl_rl/underwater_cave_multinav"
FLASH_ROOT = ROOT / "logs/flash_sac/underwater_cave_multinav"


def transition_budget(requested, ppo_envs, flash_envs):
    if requested <= 0 or ppo_envs <= 0 or flash_envs <= 0 or ppo_envs % 3 or flash_envs % 3:
        raise ValueError("Positive budget and environment counts divisible by three are required")
    quantum = math.lcm(ppo_envs * 64, flash_envs)
    return math.ceil(requested / quantum) * quantum


def unique_run(root, name):
    matches = sorted(path for path in root.glob(f"*_{name}") if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one run for {name}: {matches}")
    return matches[0]


def latest_checkpoint(run, prefix, directory=False):
    matches = [path for path in run.glob(f"{prefix}_*")
               if path.is_dir() == directory and path.stem.rsplit("_", 1)[-1].isdigit()]
    if not matches:
        raise FileNotFoundError(f"No checkpoints in {run}")
    return max(matches, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo_envs", type=int, required=True)
    parser.add_argument("--flash_envs", type=int, required=True)
    parser.add_argument("--eval_envs", type=int, default=48)
    parser.add_argument("--transitions", type=int, default=2000000)
    parser.add_argument("--episodes_per_scene", type=int, default=30)
    parser.add_argument("--flash_batch_size", type=int, default=1024)
    parser.add_argument("--flash_replay_capacity", type=int, default=131072)
    parser.add_argument("--ppo_checkpoint", type=Path,
                        default=PPO_ROOT / "2026-09-08_23-41-19_multicave48_resume_stage1/model_607.pt")
    parser.add_argument("--flash_checkpoint", type=Path,
                        default=FLASH_ROOT / "2026-09-09_00-38-10_native_ff_12env_pilot/step_0004096")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    budget = transition_budget(args.transitions, args.ppo_envs, args.flash_envs)
    if args.eval_envs <= 0 or args.eval_envs % 3 or args.episodes_per_scene <= 0:
        parser.error("Evaluation needs positive counts and environments divisible by three")
    if not args.ppo_checkpoint.is_file() or not (args.flash_checkpoint / "replay_buffer.pt").is_file():
        parser.error("Both source checkpoints must exist; FlashSAC requires saved replay")
    if args.ppo_checkpoint.resolve().parent.parent != PPO_ROOT.resolve():
        parser.error("The maintained PPO trainer resolves checkpoints under this experiment root")
    # One queue per workspace. This does not affect unrelated applications.
    lock_path = ROOT / "outputs/dual_navigation_training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        name = f"dual4060_{datetime.now():%Y%m%d_%H%M%S}"
        plan = {
            "name": name, "additional_transitions_per_algorithm": budget,
            "ppo_envs": args.ppo_envs, "flash_envs": args.flash_envs, "eval_envs": args.eval_envs,
            "ppo_iterations": budget // (args.ppo_envs * 64), "flash_vector_steps": budget // args.flash_envs,
            "flash_batch_size": args.flash_batch_size, "flash_replay_capacity": args.flash_replay_capacity,
            "flash_tf32": True,
            "episodes_per_scene": args.episodes_per_scene, "seed": 42, "domain_randomization": False,
            "ppo_resume": str(args.ppo_checkpoint.resolve()), "flash_resume": str(args.flash_checkpoint.resolve()),
            "comparison": "continued_checkpoints_with_different_architectures_and_prior_budgets",
            "resource_limits": {"host_available_mb_min": 1200, "gpu_memory_mb_max": 7650,
                                "gpu_temperature_c_max": 85, "persistent_samples": 3},
        }
        source_files = ["scripts/train.py", "scripts/train_flash_sac.py", "scripts/supervise_training.py",
                        "scripts/run_dual_navigation_training.py", "configs/learning/flash_sac_caves.yaml",
                        "source/isaac_underwater/isaac_underwater/learning/flash_sac.py",
                        "source/isaac_underwater/isaac_underwater/tasks/direct/pointnav_env.py"]
        plan["source_sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files}
        write_status(directory / "plan.json", plan)
        status = {"state": "running", "supervisor_pid": os.getpid(), "current_stage": None,
                  "jobs": {}, "plan": str(directory / "plan.json")}

        def job(stage, command, max_seconds=7200):
            status.update(current_stage=stage, live_job_status=str(directory / f"{stage}.status.json"))
            write_status(directory / "status.json", status)
            print(f"dual4060: starting {stage}; log={directory / (stage + '.log')}", flush=True)
            result = run_guarded(command, directory / f"{stage}.log", max_seconds=max_seconds)
            status["jobs"][stage] = result
            write_status(directory / "status.json", status)
            if result.get("stop_reason") == "user_or_supervisor_interrupt":
                raise KeyboardInterrupt("Queue stopped; no further training jobs will start")
            return result["state"] == "complete"

        def gate(algorithm, metrics):
            checked = subprocess.run([PYTHON, "scripts/check_navigation_metrics.py", str(metrics),
                                      "--expected-profile", "train_all"], cwd=ROOT,
                                     capture_output=True, text=True, timeout=60)
            (directory / f"{algorithm}_acceptance.log").write_text(checked.stdout + checked.stderr, encoding="utf-8")
            status["jobs"][f"{algorithm}_acceptance"] = {
                "passed": checked.returncode == 0, "return_code": checked.returncode, "metrics": str(metrics),
            }
            # Failed policy acceptance must not prevent the other requested algorithm from running.
            write_status(directory / "status.json", status)

        max_eval_steps = math.ceil(args.episodes_per_scene / (args.eval_envs // 3)) * 6000 + 100
        ppo_name, flash_name, flash_eval_name = f"{name}_ppo", f"{name}_flash", f"{name}_flash_eval"
        try:
            if not job("sensor_contract", [PYTHON, "-u", "scripts/smoke_cave_navigation.py",
                                           "--num_envs", "3", "--domain_randomization",
                                           "--headless", "--device", "cuda:0"], max_seconds=180):
                raise RuntimeError("Sensor/randomization preflight failed; training was not started")
            ppo_ok = job("ppo_train", [
                PYTHON, "-u", "scripts/train.py", "--task", TASK, "--cave_dataset_profile", "train_all",
                "--navigation_curriculum", "--num_envs", str(args.ppo_envs), "--max_iterations", str(plan["ppo_iterations"]),
                "--resume", "--load_run", args.ppo_checkpoint.parent.name, "--checkpoint", args.ppo_checkpoint.name,
                "--run_name", ppo_name, "--seed", "42", "--headless", "--enable_cameras", "--device", "cuda:0",
                "agent.save_interval=25", "env.domain_randomization_enabled=False",
            ])
            if ppo_ok:
                run = unique_run(PPO_ROOT, ppo_name)
                checkpoint = latest_checkpoint(run, "model")
                metrics = run / "evaluation_full.json"
                if job("ppo_eval", [PYTHON, "-u", "scripts/evaluate_ppo.py", "--task", TASK,
                                    "--checkpoint", str(checkpoint), "--cave_dataset_profile", "train_all",
                                    "--num_envs", str(args.eval_envs), "--episodes_per_scene", str(args.episodes_per_scene),
                                    "--max_rollout_steps", str(max_eval_steps), "--output", str(metrics),
                                    "--seed", "42", "--headless", "--device", "cuda:0"], max_seconds=3600):
                    gate("ppo", metrics)
            flash_ok = job("flash_train", [
                PYTHON, "-u", "scripts/train_flash_sac.py", "--checkpoint", str(args.flash_checkpoint.resolve()),
                "--resize_resume", "--num_envs", str(args.flash_envs), "--train_steps", str(plan["flash_vector_steps"]),
                "--replay_capacity", str(args.flash_replay_capacity), "--batch_size", str(args.flash_batch_size),
                "--save_interval_steps", "4096", "--save_replay_checkpoints", "--eval_episodes_per_scene", "0",
                "--tf32",
                "--run_name", flash_name, "--seed", "42", "--headless", "--device", "cuda:0",
            ])
            if flash_ok:
                run = unique_run(FLASH_ROOT, flash_name)
                checkpoint = latest_checkpoint(run, "step", directory=True)
                if job("flash_eval", [
                    PYTHON, "-u", "scripts/train_flash_sac.py", "--checkpoint", str(checkpoint), "--train_steps", "0",
                    "--num_envs", str(args.eval_envs), "--eval_episodes_per_scene", str(args.episodes_per_scene),
                    "--max_eval_steps", str(max_eval_steps), "--run_name", flash_eval_name,
                    "--seed", "42", "--headless", "--device", "cuda:0",
                ], max_seconds=3600):
                    evaluated = unique_run(FLASH_ROOT, flash_eval_name)
                    metrics = run / "evaluation_full.json"
                    if metrics.exists():
                        raise FileExistsError(metrics)
                    shutil.copy2(evaluated / "evaluation_full.json", metrics)
                    gate("flash", metrics)
            expected = ("ppo_train", "ppo_eval", "flash_train", "flash_eval")
            status["state"] = "complete" if all(status["jobs"].get(key, {}).get("state") == "complete" for key in expected) else "completed_with_job_failures"
        except KeyboardInterrupt:
            status["state"] = "interrupted"
        except BaseException as error:
            status.update(state="failed", error=str(error))
            traceback.print_exc()
        finally:
            status["current_stage"] = None
            write_status(directory / "status.json", status)
            print(f"dual4060: {status['state']}; status={directory / 'status.json'}", flush=True)
        return 0 if status["state"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
