# RTX 4060: sequential PPO and FlashSAC continuation

The 2026-09-09 queue **finished**, with both algorithms completing their
additional transition budgets. Its durable plan and final state are under
`outputs/dual4060_continuation_v1/`. Both policies failed entrance-to-exit
acceptance: PPO and FlashSAC each achieved **0/90** successful full episodes.
Workflow completion is not policy acceptance. The next diagnostic/repair
experiment is described in [NAVIGATION_REPAIR.md](NAVIGATION_REPAIR.md).

Final full-route collision rates were 60/90 for PPO and 16/90 for FlashSAC;
the other failures were route-tube bounds violations. FlashSAC's roughly
31.4% training success is a mixed reverse-curriculum statistic, not full-route
success. Training GPU memory peaked at 5,344 MiB (PPO) and 6,872 MiB (FlashSAC),
and the sampled peak temperature was 65 C, with no resource safety stops.

## Measured operating point

Hardware: RTX 4060 Laptop GPU, 8,188 MiB VRAM; about 15.4 GiB host RAM. No
overclock, GPU power-limit change, CPU governor change or unrelated process
termination is performed. One simulator job owns the GPU at a time.

PPO calibration resumed the same checkpoint for each candidate:

| Environments | Observation path | Transitions | Training seconds | Transitions/s |
| --- | --- | --- | --- | --- |
| 48 | Legacy packets | 24,576 | 33.87 | 726 |
| 72 | Legacy packets | 36,864 | 50.87 | 725 |
| 96 | Legacy packets | 36,864 | 50.30 | 733 |
| 96 | Batched nominal sensors | 36,864 | 47.32 | 779 |

These short measurements exclude simulator startup and are not convergence
benchmarks. Larger environment counts alone reached a throughput plateau.
The nominal observation path previously materialized one Python sensor packet
per robot, synchronized scalar GPU metadata, then stacked the tensors back
into a batch. It now reuses the same batched sensor processing directly.
The packet interface remains available for VIO, logging and domain randomization.
The fast path is limited to nominal ground-truth-localized visual observations.

Real-simulator equivalence validation compared 49 frames across six workers,
including six automatic resets, with **rtol=atol=0** for both policy and critic
observations. No rewards, actions, dynamics or network structures were changed
by this optimization. Disable it with the task config option
`batched_visual_observation_enabled=False` to reproduce the legacy path.

The 96-worker PPO test peaked at 5,002 MiB GPU memory and 64 C; host available
memory stayed above 3,312 MiB. FlashSAC at 96 workers, batch 1,024 and replay
131,072 ran around 800 transitions/s before checkpoint I/O. The TF32/resume/
periodic-replay-save smoke peaked at 6,863 MiB GPU memory and 62 C. TF32 is
enabled explicitly for the long FlashSAC job, following upstream's training
entry point; storage remains float32, with AMP and compilation disabled.
Short-run startup and frequent smoke checkpoint writes distort average GPU
utilization; do not present a sampled peak as sustained utilization.

## Finite training plan

Both algorithms receive **2,002,944 additional transitions**, rounded up from
two million to complete PPO vector rollouts:

- PPO: 96 environments, 326 additional updates, resumed from
  `2026-09-08_23-41-19_multicave48_resume_stage1/model_607.pt`.
- FlashSAC: 96 environments, 20,864 additional vector steps, resumed from
  `2026-09-09_00-38-10_native_ff_12env_pilot/step_0004096`; replay grows from
  32,768 to 131,072 and batch size changes from 256 to 1,024.
- Both use the three-cave `train_all` profile, seed 42, nominal conditions,
  reverse curriculum and the unchanged route/contact termination rules.
- PPO retains CNN+GRU and FlashSAC retains the native feed-forward visual MLP.
  Different architectures and existing training histories mean this is a
  continuation comparison, not a controlled algorithm-only experiment.

The queue first checks the randomized sensor/scene contract, then runs PPO
training, PPO full evaluation, FlashSAC training and FlashSAC full evaluation.
Each full evaluation uses 48 workers, 30 episodes per cave, actual entrance
spawns, a 300 s horizon and fixed per-worker quotas. An unsuccessful policy
acceptance gate does **not** prevent the other algorithm's requested run.

PPO checkpoints every 25 updates. FlashSAC checkpoints every 4,096 vector
steps and saves replay at each checkpoint. The expected replay file footprint
is roughly 1.54 GiB per snapshot; sufficient disk space was checked before
launch. Final metrics are placed in each training run's `evaluation_full.json`.

The supervisor samples resources every two seconds. Three consecutive unsafe
samples (host available RAM below 1,200 MiB, GPU memory above 7,650 MiB, or GPU
temperature at least 85 C) stop only its owned child process group. It first
sends interrupt, then terminates/kills that same group if shutdown hangs.
Training jobs have a two-hour wall-time cap and evaluations a one-hour cap.
Recovery uses the last saved checkpoint, not unsaved in-flight simulator state.

## Status and control

Read `outputs/dual4060_continuation_v1/status.json` for the active stage, and
`<stage>.status.json` for its child PID, live memory/temperature and final return
code. `<stage>.log` contains the actual training output;
`<stage>.resources.csv` contains the time series. `plan.json` records the
source checkpoint paths, budgets, settings and hashes of key source files.

```bash
tmux attach -t cave_dual4060
```

To stop this queue and its active child without starting the next job:

```bash
tmux send-keys -t cave_dual4060 C-c
```

The existing PPO session and old checkpoints are not modified or removed.
The queue persists independently of the current chat turn. Check status/logs
for completion; no automatic future chat notification is implied.

To create a new queue, choose a new, nonexistent output directory:

```bash
./.venv/bin/python scripts/run_dual_navigation_training.py \
  --ppo_envs 96 --flash_envs 96 --eval_envs 48 \
  --transitions 2000000 --episodes_per_scene 30 \
  --output_dir outputs/dual4060_continuation_v2
```

The lock prevents multiple copies of this queue running in the workspace.
It does not prevent unrelated user-started GPU jobs; do not start a second
simulator while this queue is using the GPU.
