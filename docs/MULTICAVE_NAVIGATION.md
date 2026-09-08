# Multi-cave entrance-to-exit navigation

`Isaac-Underwater-Cave-Navigation-v0` trains one recurrent PPO policy across
the selected caves in `caves_difficulty_v01`. Each Isaac environment is bound
to one cave, environments are assigned by deterministic round robin, and all
rollouts update the same actor and critic weights.

The task is an end-to-end visual local navigator: it emits body velocity, yaw,
and light commands. It does not currently emit a waypoint graph or a classical
planner path.

## Dataset setup

The downloaded tree is expected at
`$ISAAC_UNDERWATER_ASSET_ROOT/caves_difficulty_v01`. On this workstation the
default root is `/home/kenton/Downloads/assets`. The archive was checksum
verified before use, but it does not contain a license file; keep the mesh and
texture binaries outside git until redistribution rights are established.

Run the pure-data contract and convert a selected profile:

```bash
export ISAAC_UNDERWATER_ASSET_ROOT=/home/kenton/Downloads/assets
PYTHONPATH=source/isaac_underwater ./.venv/bin/python \
  scripts/test_cave_dataset.py
./.venv/bin/python scripts/convert_cave_asset.py \
  --config worlds/caves_difficulty_v01.yaml --profile train_all \
  --headless --device cuda:0
```

The archive does not include the reduced collision meshes mentioned by its
audit metadata. Conversion therefore uses each visual OBJ as an explicit
high-poly triangle-collision fallback. This is correct enough for the current
contract tests, but it costs memory and simulation throughput and should be
replaced when dedicated collision exports are available.

The three geometric tiers are:

| Scene | Supplied main route | Geometry |
| --- | ---: | --- |
| easy | 50.4 m | wide, horizontal, two gentle bends |
| medium | 89.7 m | four turns, one chamber, slope up to 6 degrees |
| hard | 113.4 m | four right-angle turns, two dead ends, one bottleneck, slope up to 14 degrees |

The loaded entrance-to-exit A* polylines include exterior connectors and have
reference lengths of approximately 63.49, 113.85, and 133.31 m. They are
collision-checked reference routes, not exact continuous-space geodesics.

## Policy and reward contract

The recurrent actor receives 1553 values:

- low-resolution stereo RGB and depth;
- IMU acceleration and angular velocity;
- pressure depth and the previous six-dimensional action;
- the body-frame relative exit vector and exit distance.

The relative exit is a navigation command, not a route waypoint. It is needed
to disambiguate mission intent at branches; a goal-free explorer cannot know
which opening is the requested exit on an unseen topology. The actor never
receives cave identity, difficulty, the A* centerline, route chainage, or a
privileged map.

The 19-value critic uses the compact privileged navigation state during
training. A stereo CNN encodes images and a 128-state GRU supplies memory for
turns, occlusions, and branch decisions.

Training reward combines continuous projected progress along the supplied reference route,
a soft route-deviation penalty, weak direct exit-distance progress, action
cost, collision/out-of-bounds penalties, and a terminal exit bonus. The route
is therefore a training teacher and evaluator reference only. Success means
reaching within 1 m of the route's outside-exit endpoint; contact, excessive
route deviation, and workspace escape terminate an episode.

The 300 s horizon allows cautious turns: a privileged reference follower needs
about 81, 139, and 172 s on easy, medium, and hard with the actual thruster and
drag model. All three physical traversals are collision-free. Run the geometry
and controller check independently of PPO:

```bash
./.venv/bin/python scripts/smoke_cave_route_following.py --headless --device cuda:0
```

This controller receives the full reference route and is not a learned policy
or generalization result. Cave worlds deliberately have no global `z=0` seabed:
that plane intersected the entrance geometry and produced false initial
contacts.

## Contract gate and training

Run the complete finite wiring gate:

```bash
MAX_ITERATIONS=1 NUM_ENVS=6 RUN_NAME=multicave_contract \
  ./scripts/run_multicave_navigation_rl_pipeline.sh
```

It checks source metadata, converts the selected assets, verifies distinct USD
references and safe entrance/exit semantics in parallel, runs PPO, reloads the
checkpoint, and performs a two-second balanced evaluation. One iteration is
only an integration test and cannot produce a useful navigator.

The focused contact-reset regression deliberately drives the hard-scene robot
into a wall, then requires the automatically reset episode to remain alive
with zero contact force:

```bash
./.venv/bin/python scripts/smoke_cave_navigation.py \
  --num_envs 3 --profile train_all --collision_reset_steps 240 \
  --headless --device cuda:0
```

The cave ContactSensor retains one history sample so Isaac Lab refreshes it on
every physics substep. Together with clearing the lazy sensor's reset flag,
this prevents a pre-reset GPU triangle contact from causing repeated one-step
episodes on the pinned Isaac Lab 2.3 stack.

A first full experiment can use:

```bash
MAX_ITERATIONS=2000 NUM_ENVS=6 NAVIGATION_CURRICULUM=1 \
RUN_NAME=multicave_full \
  ./scripts/run_multicave_navigation_rl_pipeline.sh
```

Six environments give two copies of each train-all scene and match the three
recurrent PPO minibatches. Increase environment count only in multiples of
three and after checking RTX memory and steps/second. Domain randomization
resamples dynamics, actuation, current, image degradation, IMU, and timing
parameters per environment at reset.

`NAVIGATION_CURRICULUM=1` enables a training-only reverse curriculum. Each cave
starts 4 m from its exit. After at least 7 successes in a sliding window of 10
episodes at its current frontier, that cave's remaining distance grows by 1.5,
up to its complete entrance-to-exit route. Caves advance independently. Old
episodes that finish after a frontier change do not promote the new frontier.
Short episodes use a shorter time limit (`10 + 4 * distance` seconds, capped at
300 s). The actual exit command stays unchanged and route annotations remain
hidden from the actor. Curriculum successes must not be reported as full-route
successes. The evaluator always disables the curriculum and uses true entrance
spawns.

The curriculum can also be enabled directly with `scripts/train.py
--navigation_curriculum`. Its frontier is environment state, not policy weights;
resuming PPO currently restarts the curriculum unless a different
`env.navigation_curriculum_initial_distance_m` is supplied. Start with nominal
conditions, then enable `DOMAIN_RANDOMIZATION=1` to assess learning under the
configured dynamics and sensor variation.

## Balanced evaluation

Use completed episodes per scene rather than a global episode count, otherwise
short failure episodes can bias the aggregate toward one cave:

```bash
./.venv/bin/python scripts/evaluate_ppo.py \
  --task Isaac-Underwater-Cave-Navigation-v0 \
  --checkpoint logs/rsl_rl/underwater_cave_multinav/<run>/model_<iteration>.pt \
  --cave_dataset_profile train_all \
  --num_envs 6 --episodes_per_scene 30 \
  --output logs/rsl_rl/underwater_cave_multinav/<run>/evaluation.json \
  --headless --device cuda:0

./.venv/bin/python scripts/check_navigation_metrics.py \
  logs/rsl_rl/underwater_cave_multinav/<run>/evaluation.json \
  --expected-profile train_all
```

The default gate requires every selected scene to have at least 30 episodes,
success rate at least 0.80, collision rate at most 0.15, and mean SPL at least
0.50. These are initial engineering acceptance thresholds, not a scientific
claim. Report per-scene values and confidence intervals for final experiments,
and repeat evaluation with `--domain_randomization`.

## Geometry generalization

`train_all` is useful for fitting a final policy to the available operating
set, but it provides no evidence of performance on unseen caves. There are
only three related generated cave identities, so an ordinary independent
train/validation/test split would be misleading. Use all three leave-one-out
runs as a minimum geometry-generalization check:

```bash
PROFILE=loo_hard_train MAX_ITERATIONS=2000 NUM_ENVS=6 NAVIGATION_CURRICULUM=1 \
DOMAIN_RANDOMIZATION=1 RUN_NAME=loo_hard \
  ./scripts/run_multicave_navigation_rl_pipeline.sh

./.venv/bin/python scripts/evaluate_ppo.py \
  --task Isaac-Underwater-Cave-Navigation-v0 \
  --checkpoint logs/rsl_rl/underwater_cave_multinav/<loo-hard-run>/model_<iteration>.pt \
  --cave_dataset_profile loo_hard_eval \
  --num_envs 3 --episodes_per_scene 30 --domain_randomization \
  --output logs/rsl_rl/underwater_cave_multinav/<loo-hard-run>/heldout_hard.json \
  --headless --device cuda:0
```

Repeat for easy and medium. The pipeline's final two-second evaluation uses
the training profile only and remains a wiring check; the held-out command is
the actual test. A defensible claim that the planner works in “most caves”
will require substantially more independent cave geometries, seeds, topology
families, appearance families, and preferably real or scanned environments.

## Current evidence

The following contracts pass on the RTX 4060 workstation:

- all three source-route and checksum contracts;
- visual/collision USD conversion for easy, medium, and hard;
- heterogeneous three-scene spawning with the expected USD in every env;
- safe entrance spawn and forced exit success/SPL semantics;
- hard-wall contact followed by a clean, non-terminal automatic reset;
- continuous segment progress, per-cave curriculum promotion, and safe promoted spawns;
- full collision-free reference-controller traversal of all three scenes;
- nominal and forced-frame-drop domain-randomized visual observations;
- one-update recurrent PPO training, checkpoint loading, and balanced metrics.

The 100-update nominal baseline `2026-09-08_19-17-27_multicave_nominal_sanity_v2`
completed 38,400 transitions after the contact-reset fix. Its full-route check
completed three episodes per scene with 0/9 successes, 2/9 collisions, and a
mean travelled path of 3.81 m. This baseline used nearest-vertex progress and
no curriculum. The evaluator clears recurrent memory between episodes;
training episode logs now publish only when a new episode finishes.

No converged checkpoint or held-out success result has been established.
