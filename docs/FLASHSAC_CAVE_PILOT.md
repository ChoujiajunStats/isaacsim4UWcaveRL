# FlashSAC multi-cave compatibility pilot

This is a finite experiment, **not a converged navigator or evidence that
FlashSAC outperforms PPO**. The original PPO trainer and checkpoints remain
unchanged. Both algorithms use the existing physics, thrusters, stereo
observations, rewards, route constraints, and per-cave reverse curriculum.

## Intended policy and the actual task

The deployment target is one shared, goal-conditioned visual navigation
policy. It consumes stereo RGB/depth, IMU, pressure, previous actions and an
exit-relative command; it emits body-frame velocity, yaw-rate and two light
commands. The existing low-level controller handles thruster allocation.
It does not output a global waypoint route. Ground-truth localization supplies
the goal-relative command in this nominal simulation experiment; using VIO
or another real localization source is a separate validation step.

The intended final visual policy needs temporal information: continue around
corners even when the exit is occluded, avoid contact, handle disturbed poses,
and reach the exit on unseen geometries. Unknown-exit exploration, mapping,
branch selection and backtracking are a larger mission than the current
known-goal task. Neither an algorithm change nor success on three related
meshes establishes that capability.

Training may use privileged route progress for shaping, but the actor does
not receive the reference route or scene identity. A current limitation is
that leaving the reference-route tube by more than 2.25 m ends the episode,
even without a physical collision. Thus an out-of-bounds failure is **not**
proof of hitting a wall. The tube constraint may penalize a collision-free
alternative path and should be reviewed against the intended deployment task.
This pilot retains it to keep the environment unchanged.

## Upstream and compatibility boundaries

Sources: [official repository](https://github.com/Holiday-Robot/FlashSAC),
[project](https://holiday-robot.github.io/FlashSAC/),
[RSS paper entry](https://www.roboticsproceedings.org/rss22/p099.html).

The bridge uses upstream revision
`87edc9061150ae9e962dd84e6544e27a1554b3ab` from `.deps/FlashSAC` without changing
its agent, networks, losses, replay implementation, or weight normalization.
The checkout is ignored, like the other local simulator dependencies. If
missing, clone the official repository there and check out this revision.
Do not run upstream `uv sync` inside the working Isaac environment: its full
environment includes other simulators and a different PyTorch stack.

The pinned upstream imports JAX only for array type aliases on this code path.
When JAX is absent, the bridge provides `flash_rl.types` NumPy/Torch aliases;
it does not pretend that JAX itself is installed. This experiment keeps the
existing Python 3.11 / torch 2.7.0+cu128 stack, uses eager float32, and disables
both configurable and helper-level compilation. It is not a benchmark of
FlashSAC's compiled throughput.

Important differences from the recurrent PPO baseline:

| Item | Existing PPO | This native FlashSAC pilot |
| --- | --- | --- |
| Actor | Stereo CNN + GRU | Flattened visual input + upstream feed-forward MLP |
| Actor input | 1,553 values | Same 1,553 values |
| Critic input | 19 privileged values | 1,553 actor values + 19 privileged values |
| Memory in policy | GRU | None |
| Data use | On-policy rollouts | Uniform off-policy replay |

This isolates compatibility, not the causal effect of replacing PPO with
FlashSAC. A fair algorithm comparison must match the visual encoder/memory,
information supplied to the critic, seeds, transition budgets, and evaluation
protocol. Report wall time and gradient updates as well as transitions;
one vector step is not one PPO iteration.

With actor+critic observation width 1,572, six actions and one-step targets,
float32 replay needs `capacity * (2 * 1572 + 6 + 3) * 4` bytes. The pilot's
32,768-transition replay is 394.125 MiB; one million transitions would need
about 11.75 GiB **for replay alone**. Network, renderer and simulator memory
are additional. Do not use upstream's million-transition GPU default on this
8 GiB GPU with these image observations.

The custom environment wrapper is necessary because the upstream Isaac
wrapper disables cameras in headless mode and supplies already-reset
observations as terminal observations. Here cameras remain enabled; the
wrapper captures observations immediately before automatic reset. Replay
receives those final observations, while the next action receives the new
episode's reset observation. True termination and time-limit truncation stay
separate, so timeout bootstrapping cannot cross into the next episode.
Only nominal sensors, ground-truth localization and one-step replay are
currently validated. PPO optimization and task semantics are unchanged; the
later shared batched-sensor fast path is validated against the legacy packet
observations with zero tolerance.

## Running finite experiments

CPU checks execute the real upstream agent, gradient updates, actor/critic
input separation, terminal-observation handling and native checkpoint/replay
round trips:

```bash
./.venv/bin/python scripts/test_flash_sac.py
```

Small simulator wiring check (not a navigation acceptance test):

```bash
./.venv/bin/python scripts/train_flash_sac.py \
  --num_envs 3 --train_steps 256 --warmup_transitions 24 \
  --replay_capacity 1024 --batch_size 64 --eval_episodes_per_scene 0 \
  --save_replay --run_name wiring_smoke --headless --device cuda:0
```

Finite 49,152-transition training experiment plus nine true-entrance episodes:

```bash
./.venv/bin/python scripts/train_flash_sac.py \
  --num_envs 12 --train_steps 4096 --eval_episodes_per_scene 3 \
  --save_replay --run_name native_ff_12env_pilot --headless --device cuda:0
```

Outputs are separate from PPO under
`logs/flash_sac/underwater_cave_multinav/<timestamp>_<run_name>/`:

- `training_summary.json`: completed training episodes and curriculum state;
- `step_<vector_steps>/`: native model/optimizer state plus metadata;
- the final checkpoint also includes replay when `--save_replay` is set;
- `evaluation_full.json`: full-horizon, curriculum-disabled, balanced per-scene
  metrics and individual episode records;
- TensorBoard events: update metrics, recent completed episodes and frontiers.

To continue training, pass the final **FlashSAC directory** to `--checkpoint`,
the same environment count and curriculum setting by default, and the number of
additional vector steps. Saved replay is required; intermediate model-only
checkpoints are deliberately rejected for training resume. Replay settings
come from checkpoint metadata and cannot silently change. This restores
optimizer/replay/curriculum but starts new simulator episodes; it is not
bit-exact resumption of simulator state or RNG state.

For larger jobs, `--resize_resume` explicitly permits changing worker count,
growing replay capacity and changing batch size. The bridge repacks a grown
ring buffer chronologically and re-creates the per-worker partial reward
returns; old replay samples and global reward statistics are retained. Total
transition accounting preserves the previous worker count's contribution.
Shrinking the buffer is rejected. `--save_replay_checkpoints` also saves replay
at intermediate checkpoints, and `--tf32` opts into the upstream entry point's
TF32 matmul mode. That mode is saved in metadata and restored for evaluation.
See [the 4060 continuation plan](DUAL_4060_TRAINING.md) for measured settings.

For evaluation only, use `--checkpoint <directory> --train_steps 0
--eval_episodes_per_scene 30`. This can select a held-out dataset profile; only
training resume requires matching scene identities/order. The evaluation
budget must allow the requested per-worker episode quotas and 300 s horizon.
Incomplete evaluations fail and are saved separately, never as a full result.

## Acceptance and next comparisons

### Measured first pilot, 2026-09-09

Run: `2026-09-09_00-38-10_native_ff_12env_pilot`. Twelve environments collected
49,152 transitions and performed 4,011 gradient updates in about 204 s
(training/checkpoint time, excluding simulator startup and full-route
evaluation). Sampled total GPU usage was 3,973 MiB (about 3.88 GiB) on the RTX 4060.
All update metrics remained finite. All 146 completed training episodes had
their pre-reset terminal observation captured, including timeouts; the final
checkpoint reloaded with exactly matching deterministic actions.

Only 8/146 training episodes succeeded at the initial 4 m frontier. No cave
advanced its curriculum. Full-route deterministic evaluation completed the
balanced nine episodes with **0/9 successes, 0/9 collisions, 9/9 route-bounds
failures, and SPL 0**. These policies typically travelled about 2.2–2.3 m
before leaving the route tube. The normal acceptance checker correctly
rejected the result for success/SPL and insufficient per-scene sample count.

Therefore: integration works, but this small-budget feed-forward configuration
has not learned the navigation task. It supplies no basis for claiming an
advantage over the much longer-trained CNN+GRU PPO, or for ruling out a better
configured FlashSAC. Both final checkpoints and replay are retained locally.

### Acceptance targets

The initial acceptance gate is at least 30 complete entrance-to-exit episodes
per selected scene, each scene success >= 80%, collision <= 15%, and mean
SPL >= 0.5. The nine-episode pilot is intentionally below the sample-count
gate, even if all nine episodes were to succeed. Curriculum success and
increasing return do not satisfy this gate.

After nominal full-route success, perform leave-one-cave-out tests with no
training updates on the held-out cave, then add pose/appearance/current and
localization disturbance tests. “Works in most scenes” ultimately needs more
independent geometry/topology families than this three-scene asset pack.

Main issues to investigate before scaling:

- Partial observability and long-horizon credit: data replay is not recurrent
  memory, and a feed-forward visual MLP is a weak final architecture here.
- Curriculum transitions: PPO's frontier moves outward in 1.5x jumps without
  explicit easier-start rehearsal; stalled promotion and forgetting are
  hypotheses to test, not established explanations of all failures.
- Route-tube constraints versus safe alternative paths; inspect trajectories
  instead of counting every out-of-bounds result as collision.
- Visual/goal feature encoding and normalization: identical input dimensions
  do not make native FlashSAC's MLP equivalent to PPO's CNN+GRU.
- Off-policy terminal handling, replay footprint and reward scaling: these
  are explicit integration risks, not navigation behaviors.
