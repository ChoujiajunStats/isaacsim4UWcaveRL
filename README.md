# Isaac Underwater Research Infrastructure

This repository is a headless-first underwater robotics baseline for an RTX
4060 8 GB workstation. The first task is a BlueROV2-like open-water PointNav
environment with explicit six-axis hydrodynamics, individual thrusters,
GT/VIO-compatible interfaces, and optional RGB/depth/IMU rendering.

## Architecture

```text
Open-water world / asset plugin
          |
BlueROV rigid body + gravity + buoyancy + drag/current
          |
thruster model -> wrench allocator -> velocity controller
          |
SensorPacket (RGB, depth, IMU, pressure, GT)
          |
GroundTruthLocalizationBackend / ExternalVIOBackend
          |
NavigationObservation -> policy -> ControlCommand
```

The policy-facing contracts live in
`source/isaac_underwater/isaac_underwater/interfaces`. They do not expose
Isaac Lab objects and can be reused by a real-robot adapter.
External VIO estimates may be submitted globally or per robot with
`submit_localization_output(..., namespace="robot_000")`; the task aggregates
per-robot estimates before constructing the policy observation.

## Modes

`physics-only` is the default: no cameras, no renderer sensor buffers, and
maximum parallelism for dynamics tests and PPO. `perception` enables headless
RGB, depth, IMU, pressure, and robot-mounted lights at a deliberately small
environment count. A GUI visualization can be added with standard Isaac Lab
launcher flags for one to four environments.

## Setup

The pinned stack is Isaac Sim 5.1.0, Isaac Lab v2.3.2, Python 3.11,
PyTorch 2.7.0+cu128, and RSL-RL 3.1.2. The installation script is resumable:

```bash
./scripts/setup_isaaclab.sh
```

The first Isaac Sim launch asks for NVIDIA's Omniverse EULA. Accept it once
interactively before running the headless checks.

## Checks and runs

Pure-Python checks do not start Isaac Sim:

```bash
./.venv/bin/python scripts/test_buoyancy.py
./.venv/bin/python scripts/test_drag.py
./.venv/bin/python scripts/test_thrusters.py
./.venv/bin/python scripts/test_interfaces.py
./.venv/bin/python scripts/test_domain_gap_runtime.py
```

After the EULA is accepted, validate the GPU task:

```bash
./.venv/bin/python scripts/smoke_pointnav.py --headless --num_envs 32 --steps 200
# apply per-environment dynamics/actuator/sensor randomization
./.venv/bin/python scripts/smoke_pointnav.py --headless --num_envs 32 --steps 200 \
  --domain_randomization
```

Validate RGB/depth/IMU/lighting separately:

```bash
./.venv/bin/python scripts/smoke_perception.py \
  --headless --num_envs 8 --steps 30 --rendering_mode performance \
  --kit_args="--/validate/iommu/enabled=false"
```

On the validated workstation this passes with NVIDIA driver `580.173.02` and
produces RGB `(120, 160, 3)` plus depth `(120, 160, 1)` for all eight
environments. Driver `595.84` crashes in the Isaac Sim 5.1 RTX scene database;
see `docs/validation_report.md`. The Kit argument suppresses the IOMMU warning
dialog for unattended headless runs after IOMMU was separately ruled out.

Validate the camera-free IMU and vehicle-light path:

```bash
./.venv/bin/python scripts/smoke_imu.py --headless --num_envs 4 --steps 30 --domain_randomization
```

Measure the 8 GB operating envelope:

```bash
./scripts/benchmark_sweep.sh
```

The default sweep writes `logs/benchmarks/physics.csv` for
32/64/128/256/512 physics environments. Request a perception sweep with:

```bash
./scripts/benchmark_sweep.sh --with-perception \
  --kit_args="--/validate/iommu/enabled=false"
```

Each successful row records VRAM, GPU utilization, simulation steps/sec,
environment steps/sec, and real-time factor.

Train and resume the minimal RSL-RL PPO task:

```bash
./.venv/bin/python scripts/train.py \
  --task Isaac-Underwater-PointNav-Direct-v0 --headless --num_envs 128 \
  --max_iterations 500
./.venv/bin/python scripts/play.py \
  --task Isaac-Underwater-PointNav-Direct-v0 --headless --num_envs 32
```

The complete finite train/evaluate/checkpoint-smoke workflow is documented in
[`docs/RL_PIPELINE.md`](docs/RL_PIPELINE.md). Run a short end-to-end wiring
check with:

```bash
MAX_ITERATIONS=1 NUM_ENVS=128 EVAL_EPISODES=16 \
  ./scripts/run_rl_pipeline.sh
```

For quantitative checkpoint evaluation without the infinite player loop:

```bash
./.venv/bin/python scripts/evaluate_ppo.py \
  --checkpoint logs/rsl_rl/underwater_pointnav/<run>/model_0.pt \
  --num_envs 16 --episodes 32 --output logs/evaluation.json \
  --headless --device cuda:0
```

For a finite checkpoint load/inference check, use `scripts/smoke_ppo.py` with
the `model_0.pt` produced by a short training run.

To exercise the external-localization process boundary without installing a
full VIO stack, replay an exported sensor stream through the explicit degraded
IMU adapter:

```bash
./.venv/bin/python scripts/vio_dead_reckoning.py \
  --input logs/vio_transport_test.jsonl \
  --output logs/vio_dead_reckoning.jsonl
```

This adapter is only a transport and health-contract example; it is not
OpenVINS/ORB-SLAM and reports `tracking_status=degraded`.

For GUI visualization of the physics task, omit `--headless` and use one to
four environments with the maintained Isaac Lab player:

```bash
./.venv/bin/python scripts/play.py \
  --task Isaac-Underwater-PointNav-Direct-v0 --num_envs 1 \
  --checkpoint logs/rsl_rl/underwater_pointnav/<run>/model_0.pt
```

## Configuration

The structured configuration surface is under `configs/`:

```text
robots/     BlueROV mass, geometry, CoM/CoB, thruster layout
worlds/     open-water dimensions and future asset plugin
sensors/    physics-only and RGB/depth/IMU rates
lighting/   ambient and vehicle-mounted lights
dynamics/   water, drag, added mass, and current
vio/        external VIO transport and health fields
rl/         PointNav PPO, domain-gap ranges, and OOD split contracts
hardware/   RTX 4060 memory limits and benchmark counts
```

## Status discipline

The pure-Python dynamics, interfaces, current/appearance transforms, VIO
transport and degraded adapter, headless physics, IMU/lighting,
multi-environment, logging, and PPO checks pass on this workstation.
RGB/depth camera startup and eight-environment tiled rendering pass with the
validated NVIDIA 580 driver. The adapter is a transport/backend contract
example; it is not claimed to be a bundled OpenVINS deployment.

The current model uses PhysX rigid-body gravity plus an explicit buoyancy,
linear/quadratic drag, angular drag, current, optional diagonal added-mass,
and center-of-buoyancy restoring torque layer. It intentionally does not model
waves, particles, fluid simulation, sonar, or DVL. Cave visual/collision asset
loading is available, while route-level free-space and direct PPO validation
remain follow-up work.

GUI and camera modes attach the detailed BlueROV2 visual from
`assets/bluerov2/bluerov2_visual.usda`; physics-only headless training does not
load it. The visual source is the Apache-2.0 Stonefish BlueROV2 repository at
commit `6448383af6b7ef6083b0eac2c08102660591e318`. Collision, mass, buoyancy,
and drag continue to use the lightweight project-owned primitive model. To
regenerate the USD files from the vendored OBJ sources, run:

```bash
./.venv/bin/python scripts/convert_bluerov2_assets.py --force
```

When enabled, `domain_randomization_enabled` samples the YAML ranges per
environment and applies actuator strength/asymmetry/dead-zone/response/noise/
latency, buoyancy, drag, CoB offset, current, added-mass scaling, image-space
appearance, IMU extrinsics, and sensor timing/noise perturbations. The sampled
values are attached to each `SensorPacket` metadata record for replay. Ambient
light-scale and texture-scale values are retained in that contract but are not
written into USD on every reset, because this Isaac Sim host can hang during
stage teardown after such mutations. The localization and geometry ranges are
exposed in the same contract for the external-VIO/world plugins; their full
runtime models remain follow-up work.
The robot YAML keeps explicit CoM and inertia fields. The current
Fabric-compatible cuboid path uses the recorded uniform-box approximation;
applying a non-uniform custom USD inertia/CoM is tracked as a renderer/scene-
stack follow-up.

## Cave assets

Large cave meshes remain outside git. Point the registry at the downloaded
asset tree and convert the visual/collision pair when needed:

```bash
export ISAAC_UNDERWATER_ASSET_ROOT=/home/kenton/Downloads/assets
./.venv/bin/python scripts/convert_cave_asset.py \
  --config worlds/porth_yr_ogof_sump9.yaml --headless --device cpu
./.venv/bin/python scripts/smoke_cave_asset.py \
  --config worlds/porth_yr_ogof_sump9.yaml --num_envs 1 --steps 2 \
  --headless --device cuda:0
```

The cave stage keeps detailed visual geometry separate from the provided
collision mesh and clones both below each environment. Spawn points use the
provisional centerline metadata; they are not an entrance annotation. See
[`docs/CAVE_ASSET_PIPELINE.md`](docs/CAVE_ASSET_PIPELINE.md) for current
PASS/NOT-YET-VALIDATED results. The cave visual pilot is registered as
`Isaac-Underwater-Cave-VisualPilot-v0`. Its 1553-D stereo actor and 19-D
privileged critic pass finite train/evaluate/checkpoint-load smoke. Cave route
progress and centerline deviation are provisional teacher signals. PhysX
contact is exposed as a conservative static-collider proxy; strict cave-only
collision termination is disabled by default because GPU PhysX 5.1 cannot
filter the imported triangle collider. This is a pipeline contract, not a
convergence or real-world validation result.

The follow-on `Isaac-Underwater-Cave-Explore-v0` removes the goal command from
the actor, replaces the flattened-image MLP with a lightweight stereo CNN +
GRU, and uses label-free voxel novelty plus depth clearance. Its finite
train/evaluate/checkpoint gate is:

```bash
MAX_ITERATIONS=1 NUM_ENVS=2 EVAL_ENVS=1 EVAL_EPISODES=1 \
  ./scripts/run_exploration_rl_pipeline.sh
```

This task starts at one fixed, centerline-derived safe point and has no entry
event, so it is not an outside-to-inside entrance detector. The separate
`Isaac-Underwater-Cave-Entry-v0` task infers a portal from the automatic
skeleton-clearance transition and keeps the same no-command actor contract:

```bash
MAX_ITERATIONS=1 NUM_ENVS=2 EVAL_ENVS=1 EVAL_EPISODES=1 \
  ./scripts/run_cave_entry_rl_pipeline.sh
```

The Porth portal and finite PPO pipeline pass, but the saved one-update policy
is an untrained baseline; held-out entrance discovery remains unvalidated. See
[`docs/VISUAL_NAVIGATION_VALIDATION.md`](docs/VISUAL_NAVIGATION_VALIDATION.md).
