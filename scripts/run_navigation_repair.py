#!/usr/bin/env python3
"""Finite PPO exit-control repair, then frozen short/full evaluation.

This is an explicitly changed training recipe, not an algorithm comparison.
Original checkpoints and standard acceptance thresholds remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
from pathlib import Path
import subprocess
import traceback

from run_dual_navigation_training import PPO_ROOT, PYTHON, TASK, latest_checkpoint, unique_run
from supervise_training import ROOT, run_guarded, write_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--flash_checkpoint", type=Path, help="Optional frozen FlashSAC action diagnostic before PPO repair")
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument("--num_envs", type=int, default=96)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if (not checkpoint.is_file() or checkpoint.parent.parent != PPO_ROOT.resolve()
            or args.iterations <= 0 or args.num_envs <= 0 or args.num_envs % 3):
        parser.error("Require a PPO checkpoint under the experiment root and positive, three-way balanced workers")
    if args.flash_checkpoint is not None and not (args.flash_checkpoint / "metadata.json").is_file():
        parser.error("FlashSAC diagnostics require a checkpoint metadata file")
    lock_path = ROOT / "outputs/dual_navigation_training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory = args.output_dir.resolve()
        directory.mkdir(parents=True, exist_ok=False)
        run_name = f"exit_rehearsal_{datetime.now():%Y%m%d_%H%M%S}"
        train = [PYTHON, "-u", "scripts/train.py", "--task", TASK, "--cave_dataset_profile", "train_all",
                 "--navigation_curriculum", "--num_envs", str(args.num_envs), "--max_iterations", str(args.iterations),
                 "--resume", "--load_run", checkpoint.parent.name, "--checkpoint", checkpoint.name,
                 "--run_name", run_name, "--seed", "42", "--headless", "--enable_cameras", "--device", "cuda:0",
                 "agent.navigation_weights_only=True", "agent.navigation_reset_noise_std=0.4",
                 "agent.algorithm.entropy_coef=0.001", "agent.algorithm.learning_rate=0.0001",
                 "agent.algorithm.schedule=fixed", "agent.save_interval=24",
                 "env.navigation_curriculum_initial_distance_m=4.0", "env.navigation_curriculum_window=20",
                 "env.navigation_curriculum_success_threshold=0.8", "env.navigation_curriculum_growth=1.25",
                 "env.navigation_rehearsal_probability=0.5", "env.navigation_rehearsal_min_distance_m=2.0",
                 "env.navigation_rehearsal_max_distance_m=12.0", "env.domain_randomization_enabled=False"]
        source_files = ["scripts/run_navigation_repair.py", "scripts/train.py", "scripts/diagnose_navigation_actions.py",
                        "source/isaac_underwater/isaac_underwater/learning/navigation_runner.py",
                        "source/isaac_underwater/isaac_underwater/navigation/exit_curriculum.py",
                        "source/isaac_underwater/isaac_underwater/tasks/direct/pointnav_env.py",
                        "source/isaac_underwater/isaac_underwater/tasks/direct/agents/visual_ppo_cfg.py"]
        plan = {"run_name": run_name, "algorithm": "PPO", "source_checkpoint": str(checkpoint),
                "flash_diagnostic_checkpoint": str(args.flash_checkpoint.resolve()) if args.flash_checkpoint else None,
                "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "new_transitions": args.num_envs * 64 * args.iterations, "train_command": train,
                "changes": "weights/normalizers warm start; fresh optimizer/curriculum; short-exit rehearsal; slower promotion; lower exploration and learning rate",
                "not_changed": "actor observations/architecture; actions; dynamics; rewards; collision and bounds; full-route acceptance",
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}}
        write_status(directory / "plan.json", plan)
        state = {"state": "running", "plan": str(directory / "plan.json"), "stages": {}}

        def job(name, command, max_seconds=1800):
            state["active_stage"] = name
            write_status(directory / "status.json", state)
            result = run_guarded(command, directory / f"{name}.log", max_seconds=max_seconds)
            state["stages"][name] = result
            write_status(directory / "status.json", state)
            if result["state"] != "complete":
                raise RuntimeError(f"{name} failed/stopped: {result.get('stop_reason')} rc={result['return_code']}")

        try:
            if args.flash_checkpoint is not None:
                job("flash_diagnostic", [PYTHON, "scripts/diagnose_navigation_actions.py", "--algorithm", "flash",
                                         "--checkpoint", str(args.flash_checkpoint.resolve()),
                                         "--output", str(directory / "flash_diagnostic.json"), "--num_envs", "18",
                                         "--scopes", "short", "frontier"])
            job("ppo_train", train)
            run = unique_run(PPO_ROOT, run_name)
            model = latest_checkpoint(run, "model")
            state.update(run_directory=str(run), checkpoint=str(model))
            job("short_diagnostic", [PYTHON, "scripts/diagnose_navigation_actions.py", "--algorithm", "ppo",
                                      "--checkpoint", str(model), "--output", str(directory / "short_diagnostic.json"),
                                      "--num_envs", "18", "--scopes", "short", "--modes", "deterministic", "stochastic"])
            metrics = run / "evaluation_full.json"
            job("full_evaluation", [PYTHON, "scripts/evaluate_ppo.py", "--task", TASK, "--checkpoint", str(model),
                                     "--num_envs", "48", "--episodes_per_scene", "30", "--episode_length_s", "300",
                                     "--max_rollout_steps", "20000", "--seed", "42", "--cave_dataset_profile", "train_all",
                                     "--output", str(metrics), "--headless", "--enable_cameras"])
            result = subprocess.run([PYTHON, "scripts/check_navigation_metrics.py", str(metrics), "--expected-profile", "train_all"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=60)
            if result.returncode not in (0, 1):
                raise RuntimeError(f"Acceptance checker execution failed: {result.stderr}")
            state.update(state="complete", active_stage=None, accepted=result.returncode == 0,
                         acceptance_output=result.stdout + result.stderr, evaluation=str(metrics))
        except BaseException as error:
            state.update(state="failed", error=str(error))
            write_status(directory / "status.json", state)
            raise
        write_status(directory / "status.json", state)
        print(f"Repair experiment complete; policy accepted={state['accepted']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
