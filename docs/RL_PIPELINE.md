# RL Pipeline

The PointNav policy uses the same BlueROV2-like rigid body, controller,
thruster allocator, hydrodynamics layer, and vectorized Isaac environment for
training and evaluation. The pipeline is deliberately split into finite
commands so a failed stage leaves a usable checkpoint and a useful log.

## Train

```bash
./.venv/bin/python scripts/train.py \
  --task Isaac-Underwater-PointNav-Direct-v0 \
  --num_envs 128 --max_iterations 500 \
  --headless --device cuda:0
```

The run directory is written below
`logs/rsl_rl/underwater_pointnav/<timestamp>/`. RSL-RL writes `model_*.pt`,
the resolved environment/agent YAML, TensorBoard events, and the captured git
diff there.

## Resume

Resume from a known run and checkpoint without changing the task contract:

```bash
./.venv/bin/python scripts/train.py \
  --task Isaac-Underwater-PointNav-Direct-v0 \
  --resume \
  --load_run 2026-09-06_11-49-46 \
  --checkpoint model_0.pt \
  --num_envs 128 --max_iterations 500 \
  --headless --device cuda:0
```

`--checkpoint` is the filename inside `--load_run`; use an absolute checkpoint
path with `scripts/smoke_ppo.py` or `scripts/evaluate_ppo.py` when a run is
being moved between machines.

## Finite Evaluation

Unlike the GUI player, the evaluator exits after a fixed number of completed
episodes and writes episode-level metrics:

```bash
./.venv/bin/python scripts/evaluate_ppo.py \
  --checkpoint logs/rsl_rl/underwater_pointnav/<run>/model_0.pt \
  --num_envs 16 --episodes 32 \
  --output logs/rsl_rl/underwater_pointnav/<run>/evaluation.json \
  --headless --device cuda:0
```

The JSON includes mean return, episode length, success rate, out-of-bounds
rate, timeout rate, current mode, hydrodynamics preset, and whether domain
randomization was enabled. OOD checks can use the same evaluator, for example:

```bash
./.venv/bin/python scripts/evaluate_ppo.py \
  --checkpoint logs/rsl_rl/underwater_pointnav/<run>/model_0.pt \
  --num_envs 16 --episodes 32 --domain_randomization \
  --hydrodynamics_preset hydro_rl --current_mode slow_varying \
  --output logs/rsl_rl/underwater_pointnav/<run>/evaluation_ood.json \
  --headless --device cuda:0
```

## Export and Interactive Play

`scripts/play.py` delegates to Isaac Lab's maintained RSL-RL player. It loads
the checkpoint, exports `exported/policy.pt` and `exported/policy.onnx`, then
runs the policy. Add `--video --video_length N` for a finite visual rollout;
without video it is intentionally an interactive/infinite player.

## One-Command Regression

For a short train -> evaluate -> smoke cycle:

```bash
MAX_ITERATIONS=1 NUM_ENVS=128 EVAL_EPISODES=16 \
  ./scripts/run_rl_pipeline.sh
```

The default one-iteration setting is a wiring check, not a convergence claim.
For learning experiments, increase `MAX_ITERATIONS` and compare the saved
evaluation JSON files across nominal, randomized, and OOD conditions.

## Visual Cave PPO Contract

The current visual pilot uses the same BlueROV2 body, thruster/controller,
hydrodynamics, and PhysX scene as the state task:

```text
actor policy: 1553-D stereo RGB/depth + IMU + pressure + previous action + mission command
critic state: 19-D privileged navigation state
action:       6-D body velocity/yaw command plus left/right light scales
```

Run a finite train -> evaluate -> checkpoint-load gate with cameras enabled:

```bash
MAX_ITERATIONS=1 NUM_ENVS=2 EVAL_ENVS=1 EVAL_EPISODES=1 \
  ./scripts/run_visual_rl_pipeline.sh
```

This validates tensor shapes, camera startup, asymmetric actor/critic loading,
and finite checkpoint inference. It is not a convergence or sim-to-real claim.
The cave route reward uses a provisional centerline for teacher shaping. A
PhysX contact reporter is exposed as a conservative static-collider contact
proxy; GPU PhysX 5.1 cannot filter the imported triangle collider as
`cave-only`, so strict collision termination remains opt-in via
`cave_contact_termination_enabled`.

## Label-free recurrent exploration contract

`Isaac-Underwater-Cave-Explore-v0` is separate from the goal-conditioned
pilot. Its actor receives 1549 values (stereo RGB/depth, IMU, pressure, and
previous action), with no goal/centerline/entrance command. A lightweight
stereo CNN produces a 64-D visual latent, scalar measurements produce a 32-D
latent, and a 128-D GRU carries temporal state. The asymmetric 21-D critic is
privileged during training only.

Run the finite recurrent pipeline with:

```bash
MAX_ITERATIONS=1 NUM_ENVS=2 EVAL_ENVS=1 EVAL_EPISODES=1 \
  ./scripts/run_exploration_rl_pipeline.sh
```

Exploration reward is based on per-environment new `0.5 m` workspace voxels
and stereo forward clearance; every goal and provisional route reward is
disabled. Evaluation adds unique-voxel, bounded-workspace coverage, and path
length metrics. See `docs/VISUAL_NAVIGATION_VALIDATION.md` for the exact
observation/reward contract and current limitations.

## Geometry-inferred cave-entry contract

`Isaac-Underwater-Cave-Entry-v0` keeps the same 1549-D vision-only actor and
adds three portal-state values to the privileged critic (24 values total).
The portal is inferred from a non-finite-to-safe surface-clearance transition
at an endpoint of the automatically generated cave skeleton. There is no
hand-authored entrance point and no portal vector in the actor observation.

Run its finite gate with:

```bash
MAX_ITERATIONS=1 NUM_ENVS=2 EVAL_ENVS=1 EVAL_EPISODES=1 \
  ./scripts/run_cave_entry_rl_pipeline.sh
```

The entry reward is sparse; weak voxel novelty only supports visual search.
The Porth portal-inference and pipeline contracts pass, but the one-update
checkpoint is not trained. Success on held-out cave geometry remains to be
demonstrated.

## Shared multi-cave exit navigator

`Isaac-Underwater-Cave-Navigation-v0` binds parallel environments to the
easy/medium/hard caves in a balanced round robin while training one shared
stereo-CNN + GRU PPO policy. Run its finite end-to-end gate with:

```bash
MAX_ITERATIONS=1 NUM_ENVS=6 RUN_NAME=multicave_contract \
  ./scripts/run_multicave_navigation_rl_pipeline.sh
```

The actor receives a relative exit command but no cave identity, map, route,
or centerline. The reference route is used only for reward shaping and SPL-like
evaluation. Use `--episodes_per_scene` for balanced evaluation and the
leave-one-out profiles for unseen-geometry measurements. See
`docs/MULTICAVE_NAVIGATION.md` for training, acceptance, and limitations.
