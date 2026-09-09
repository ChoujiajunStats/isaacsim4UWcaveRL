#!/usr/bin/env python3
"""Run only our child process group with resource logging and safety limits.

No GPU power/clock changes, no machine-wide process termination. Persistent
low host memory, VRAM pressure or high temperature stops this owned job.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write_status(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def memory_fields(text):
    return {line.split(":", 1)[0]: int(line.split()[1]) / 1024 for line in text.splitlines()
            if line.startswith(("MemAvailable:", "SwapFree:", "SwapTotal:", "VmRSS:"))}


def sample_resources(pid):
    memory = memory_fields(Path("/proc/meminfo").read_text())
    try:
        rss = memory_fields(Path(f"/proc/{pid}/status").read_text()).get("VmRSS", 0.)
    except FileNotFoundError:
        rss = 0.
    result = {"host_available_mb": memory["MemAvailable"], "process_rss_mb": rss,
              "swap_used_mb": memory["SwapTotal"] - memory["SwapFree"]}
    output = subprocess.check_output([
        "nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5).strip().split(",")
    for key, value in zip(("gpu_memory_mb", "gpu_util_percent", "gpu_temperature_c", "gpu_power_w"), output):
        try:
            result[key] = float(value.strip())
        except ValueError:
            result[key] = None
    return result


@dataclass
class ResourceGuard:
    min_available_mb: float = 1200.
    max_gpu_memory_mb: float = 7650.
    max_temperature_c: float = 85.
    required_samples: int = 3
    consecutive: int = 0

    def check(self, sample):
        reasons = []
        if sample["host_available_mb"] < self.min_available_mb:
            reasons.append("low_host_available_memory")
        if sample.get("gpu_memory_mb") is not None and sample["gpu_memory_mb"] > self.max_gpu_memory_mb:
            reasons.append("gpu_memory_headroom")
        if sample.get("gpu_temperature_c") is not None and sample["gpu_temperature_c"] >= self.max_temperature_c:
            reasons.append("gpu_temperature")
        self.consecutive = self.consecutive + 1 if reasons else 0
        return ",".join(reasons) if self.consecutive >= self.required_samples else None


def signal_owned_group(process, signum):
    if process.poll() is None:
        # start_new_session=True makes this child's PID its process-group ID.
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass


def run_guarded(command, log_path, *, max_seconds=7200, guard=None, poll_seconds=2.):
    guard = guard or ResourceGuard()
    log_path = Path(log_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    status_path = log_path.with_suffix(".status.json")
    resource_path = log_path.with_suffix(".resources.csv")
    if any(path.exists() for path in (log_path, status_path, resource_path)):
        raise FileExistsError(f"Refusing to overwrite a previous supervised run: {log_path}")
    child_env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    state = {"command": command, "log": str(log_path), "resources": str(resource_path),
             "started_utc": datetime.now(timezone.utc).isoformat(), "state": "starting"}
    with log_path.open("x", encoding="utf-8") as log, resource_path.open("x", newline="", encoding="utf-8") as resource_file:
        process = subprocess.Popen(command, cwd=ROOT, env=child_env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        state.update(pid=process.pid, state="running")
        write_status(status_path, state)
        requested_stop = []
        original_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.signal(signum, lambda number, frame: requested_stop.append(number))
        started = time.monotonic()
        stopped_at = None
        stop_reason = None
        resource_writer = None
        samples = []
        probe_errors = 0
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    sample = {"elapsed_s": round(elapsed, 3), **sample_resources(process.pid)}
                    probe_errors = 0
                    samples.append(sample)
                    if resource_writer is None:
                        resource_writer = csv.DictWriter(resource_file, fieldnames=list(sample))
                        resource_writer.writeheader()
                    resource_writer.writerow(sample)
                    resource_file.flush()
                    reason = guard.check(sample)
                    state["latest_resources"] = sample
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    probe_errors += 1
                    reason = "resource_probe_unavailable" if probe_errors >= 3 else None
                    state["probe_error"] = str(error)
                if stopped_at is None:
                    if requested_stop:
                        reason = "user_or_supervisor_interrupt"
                    elif elapsed > max_seconds:
                        reason = "time_budget"
                    if reason:
                        stop_reason = reason
                        stopped_at = time.monotonic()
                        state.update(state="stopping", stop_reason=reason)
                        signal_owned_group(process, signal.SIGINT)
                elif time.monotonic() - stopped_at > 20:
                    signal_owned_group(process, signal.SIGKILL)
                elif time.monotonic() - stopped_at > 10:
                    signal_owned_group(process, signal.SIGTERM)
                state["elapsed_s"] = round(elapsed, 3)
                write_status(status_path, state)
                time.sleep(poll_seconds)
            return_code = process.wait()
        finally:
            if process.poll() is None:
                signal_owned_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    signal_owned_group(process, signal.SIGKILL)
                    process.wait()
            for signum, handler in original_handlers.items():
                signal.signal(signum, handler)
        state.update(state="complete" if return_code == 0 and stop_reason is None else "failed",
                     return_code=return_code, stop_reason=stop_reason,
                     elapsed_s=round(time.monotonic() - started, 3),
                     finished_utc=datetime.now(timezone.utc).isoformat())
        if samples:
            state["resource_summary"] = {
                "min_host_available_mb": min(row["host_available_mb"] for row in samples),
                "max_process_rss_mb": max(row["process_rss_mb"] for row in samples),
                "max_gpu_memory_mb": max((row["gpu_memory_mb"] or 0) for row in samples),
                "max_gpu_temperature_c": max((row["gpu_temperature_c"] or 0) for row in samples),
                "mean_gpu_util_percent": sum((row["gpu_util_percent"] or 0) for row in samples) / len(samples),
            }
        write_status(status_path, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max_seconds", type=float, default=7200)
    parser.add_argument("--min_available_mb", type=float, default=1200)
    parser.add_argument("--max_gpu_memory_mb", type=float, default=7650)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.max_seconds <= 0:
        parser.error("Supply a command after -- and a positive time budget")
    result = run_guarded(command, args.log, max_seconds=args.max_seconds,
                         guard=ResourceGuard(min_available_mb=args.min_available_mb,
                                             max_gpu_memory_mb=args.max_gpu_memory_mb))
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result["state"] == "complete" else 1)


if __name__ == "__main__":
    main()
