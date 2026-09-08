# Visual Cave Navigation Validation

## Scope and status

Three visual RL contracts are intentionally separate:

| Task | Actor command | Network | Reward purpose | Status |
| --- | --- | --- | --- | --- |
| `Isaac-Underwater-Cave-VisualPilot-v0` | local goal direction and distance | flattened low-resolution stereo + MLP | provisional centerline PointNav | pipeline PASS; convergence NOT YET VALIDATED |
| `Isaac-Underwater-Cave-Explore-v0` | none | stereo CNN + GRU | label-free workspace exploration | pipeline PASS; convergence NOT YET VALIDATED |
| `Isaac-Underwater-Cave-Entry-v0` | none | stereo CNN + GRU | find and cross an automatically inferred portal | pipeline PASS; convergence NOT YET VALIDATED |

The exploration task remains a coverage curriculum with a fixed,
centerline-derived spawn and no entry event.  The separate entry task infers
portal candidates from the automatic skeleton's endpoint
surface-clearance transition: a continuous non-finite exterior run must lead
to finite, collision-safe clearance.  No entrance coordinate is authored by a
human or exposed to the actor.  This inference currently passes on the Porth
asset only; generalization to arbitrary open photogrammetry meshes and learned
entry performance remain **NOT YET VALIDATED**.

## Exploration actor contract

The actor input has 1549 values:

```text
left RGB       3 x 12 x 16
right RGB      3 x 12 x 16
left depth     1 x 12 x 16
right depth    1 x 12 x 16
IMU            6
pressure       1
previous action 6
-----------------
total          1549
```

No four-value goal direction/distance command is appended.  The leakage smoke
mutates `_goal_pos_w` by a large offset without stepping or rendering and
requires both actor and critic observations to remain unchanged.

The project-owned RSL-RL policy applies:

```text
8-channel stereo tensor
  -> Conv(8,16,stride=2)
  -> Conv(16,32,stride=2)
  -> Conv(32,32)
  -> adaptive 3x4 pooling
  -> 64-D visual latent

13 scalar values -> empirical normalization -> 32-D scalar latent

[64 + 32] -> 128-D GRU -> actor MLP -> 6 actions
```

The six actions retain the validated high-level control contract:

```text
[surge, sway, heave, yaw_rate, light_left, light_right]
```

The recurrent exploration critic has 21 values and uses privileged relative position, body velocity,
angular velocity, projected gravity, previous action, workspace coverage,
forward clearance, and contact force.  It contains no goal or centerline
coordinate and is discarded for deployment.  The entry task's 24-D critic
adds normalized signed portal depth, radial gate distance, and entry state;
these three inferred geometry values never enter the actor.

## Label-free reward and metrics

Each environment owns a GPU boolean visitation grid with `0.5 m` voxels.
Entering an unvisited voxel produces reward; revisiting produces a small time
penalty.  A robust tenth-percentile depth from the central half of both stereo
views penalizes forward clearance below `0.8 m`.  Action cost, workspace bounds,
and the conservative contact penalty remain active.  Goal progress, goal bonus,
heading reward, route progress, and centerline-deviation reward are all zero or
disabled for this task.

The entry task reuses the same actor and a weak voxel-novelty prior.  It starts
1.5--2.5 m outside an inferred portal with random yaw and receives a sparse
25-point event only after crossing 0.75 m through the portal plane within a
1.5 m radial gate.  There is no dense portal direction or distance reward.

`workspace_coverage_fraction` uses every voxel in the configured bounding box
as its denominator.  It is not inferred cave free-space coverage and must not
be compared across worlds with different bounds without normalization.

## Stereo self-occlusion finding

The source-derived camera points are at body `x=0.160 m`, while the opaque
converted hull reaches approximately `x=0.222 m`.  The original `0.05 m` near
plane therefore rendered the vehicle itself and produced a false robust depth
of `0.159 m` at the spawn.  The stereo tasks now use a documented `0.25 m`
near-plane self-occlusion guard.  At the same smoke-test spawn, robust depth
became `4.496 m`.

The guard does not change the source-derived `0.145 m` baseline.  It is an
engineering workaround for an opaque visual asset, not a real camera extrinsic
or depth calibration.  A measured camera mount and transparent housing model
remain **NOT YET VALIDATED**.

## Reproducible checks

Pure PyTorch contracts:

```bash
PYTHONPATH=source/isaac_underwater ./.venv/bin/python scripts/test_exploration.py
PYTHONPATH=source/isaac_underwater ./.venv/bin/python scripts/test_cave_entry.py
PYTHONPATH=source/isaac_underwater ./.venv/bin/python scripts/test_visual_policy.py
PYTHONPATH=source/isaac_underwater ./.venv/bin/python \
  scripts/check_cave_portals.py --config worlds/porth_yr_ogof_sump9.yaml
```

Isaac task and goal-leakage gate:

```bash
PYTHONPATH=source/isaac_underwater:.deps/IsaacLab/source \
./.venv/bin/python scripts/smoke_cave_explore.py \
  --num_envs 1 --steps 2 --headless --device cuda:0
PYTHONPATH=source/isaac_underwater:.deps/IsaacLab/source \
./.venv/bin/python scripts/smoke_cave_entry.py \
  --num_envs 1 --headless --device cuda:0
```

Finite train/evaluate/checkpoint gate:

```bash
MAX_ITERATIONS=1 NUM_ENVS=2 EVAL_ENVS=1 EVAL_EPISODES=1 \
./scripts/run_exploration_rl_pipeline.sh
MAX_ITERATIONS=1 NUM_ENVS=2 EVAL_ENVS=1 EVAL_EPISODES=1 \
./scripts/run_cave_entry_rl_pipeline.sh
```

## Latest local evidence

RTX 4060 8 GB, Isaac Sim 5.1, RSL-RL 3.1.2:

```text
exploration_contract: PASS
visual_policy_contract: PASS
cave_explore_smoke: PASS
CNN/GRU PPO update: PASS (2 envs, 32 timesteps, about 23 env-steps/s)
checkpoint load/inference: PASS
finite 599-step evaluation: PASS
automatic Porth portal: PASS (chainage 4.00 m, exterior run 4.00 m, clearance 0.805 m)
entry event/reset/goal-leakage smoke: PASS
entry CNN/GRU PPO update and checkpoint inference: PASS (2 envs, 32 timesteps)
```

The one-update checkpoint is an untrained pipeline baseline:

```text
unique voxels: 10
path length: 3.218 m
workspace coverage fraction: 2.84e-5
collision proxy episode rate: 1.0
timeout rate: 1.0
```

These numbers do not establish navigation performance.  Before a convergence
claim, the project still needs multiple train seeds, held-out cave layouts,
held-out portal inference, a validated cave-only collision signal, and a long
training comparison against non-recurrent and goal-conditioned baselines.

The entry one-update checkpoint is likewise only a pipeline baseline.  With
seed 42 it collided after 123 steps: success `0`, collision `1`, spatial
out-of-bounds `0`, four visited voxels, and return `-20.04`.  Collision and
spatial out-of-bounds are logged as distinct terminal causes.
