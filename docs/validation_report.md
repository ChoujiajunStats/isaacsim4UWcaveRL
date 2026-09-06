# Validation Report

The current Fossen-compatible marine dynamics evidence chain is maintained in
[`HYDRODYNAMICS_VALIDATION.md`](HYDRODYNAMICS_VALIDATION.md) and the source
provenance in [`reference_models.md`](reference_models.md). The older baseline
results below remain useful for asset, sensor, PPO, and camera regressions; they
do not replace the parameter-validation status in the new report.

Validated on 2026-09-06 in `/home/kenton/Desktop/IsaacSim4uwRL` after accepting
the NVIDIA Omniverse EULA.

## Results

| Area | Status | Evidence |
| --- | --- | --- |
| Isaac Sim / Isaac Lab installation | PASS | `bash scripts/check_installation.sh`; Isaac Sim 5.1.0.0, Isaac Lab v2.3.2, Python 3.11.16, PyTorch 2.7.0+cu128, RSL-RL 3.1.2 |
| Headless physics task | PASS | `scripts/smoke_pointnav.py --headless --num_envs 32 --steps 200`; mean displacement 0.3863 m; rechecked after interface integration |
| BlueROV-like rigid body and scene | PASS | PointNav scene builds and runs on the RTX 4060 |
| BlueROV2 visual asset | PASS | Stonefish BlueROV2 OBJ meshes converted to `assets/bluerov2/bluerov2_visual.usda`; 10 visual meshes (hull, frame, 8 propellers), zero visual collision meshes |
| Thruster model and wrench allocation | PASS | `scripts/test_thrusters.py` |
| Buoyancy / center of buoyancy | PASS | `scripts/test_buoyancy.py` |
| Linear/quadratic drag and current | PASS | `scripts/test_drag.py`, `scripts/test_current.py`; constant/sinusoidal/random-walk current profiles |
| Velocity controller and command modes | PASS | `scripts/test_controller.py`, `scripts/test_command_modes.py`; velocity, wrench, and direct-thruster paths |
| Data contracts and localization API | PASS | `scripts/test_interfaces.py`; GT and VIO state sources are distinct, with per-robot VIO outputs aggregated into vectorized policy state |
| External VIO transport | PASS | `scripts/test_vio_transport.py`, `scripts/test_vio_dead_reckoning.py`; JSONL import/export, namespace-aware UDP receiver, and runnable degraded IMU adapter |
| Domain-gap randomization | PASS | `scripts/test_randomization.py`, `scripts/test_domain_gap_runtime.py`; 35 sampled parameters including an independent added-mass scale, runtime dynamics/actuator/sensor/appearance hooks, and packet metadata |
| Underwater appearance transform | PASS | `scripts/test_appearance.py`; attenuation/backscatter/exposure/contrast and image-space motion-blur post-render hooks |
| Episode logging | PASS | 2 env x 8 steps JSONL smoke; all required GT/estimated/health/control/thruster/sensor/collision/goal/reward fields |
| Physics parallelism | PASS | `logs/benchmarks/physics.csv`, 32/64/128/256/512 environments; all completed |
| PPO training | PASS | 1 iteration, 16 environments, 384 timesteps; `model_0.pt` produced |
| PPO checkpoint inference | PASS | `scripts/smoke_ppo.py`, 4 environments, 10 steps; finite actions/rewards |
| RGB/depth camera | PASS | NVIDIA `580.173.02`; `scripts/smoke_perception.py --headless --num_envs 8 --steps 30 --rendering_mode performance`; RGB `(120, 160, 3)`, depth `(120, 160, 1)` for all environments |
| IMU sensor runtime | PASS | `scripts/smoke_imu.py --headless --num_envs 4 --steps 8 --domain_randomization`; synchronized 3-axis IMU packet plus randomized timing/extrinsic metadata |
| Underwater lighting runtime | PASS | Same IMU smoke; ambient dome plus per-environment vehicle lights are valid after clone; distance/flicker update utility is covered by appearance checks |

## Physics benchmark

The measured physics-only envelope used the project benchmark script on
`cuda:0`:

| Environments | VRAM | Env steps/s | Sim steps/s | Real-time factor |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 932 MiB | 4551.80 | 426.73 | 7.112 |
| 64 | 932 MiB | 9091.74 | 426.18 | 7.103 |
| 128 | 932 MiB | 18106.29 | 424.37 | 7.073 |
| 256 | 932 MiB | 35625.82 | 417.49 | 6.958 |
| 512 | 948 MiB | 69998.84 | 410.15 | 6.836 |

The reported VRAM is board usage sampled during the benchmark, not a promise
that a desktop session or a camera workload fits in the same budget.

The default training preset is 128 environments. 512 physics environments were
measured successfully; 256 is a conservative long-running choice when the
desktop session or additional logging is active.

## Architecture Coverage

The active data path is:

```text
WorldAssetPlugin / OpenWaterWorld
        -> BlueROV-like rigid body + Hydrodynamics + CurrentField + domain-gap state
        -> SensorPacket (per-robot namespace and synchronized metadata)
        -> GroundTruthLocalizationBackend or ExternalVIOBackend
        -> NavigationObservation
        -> ControlCommand (velocity / wrench / thruster)
        -> command_to_thruster -> ThrusterModel
        -> PointNav / RSL-RL policy
```

Physics-only mode leaves camera buffers and vehicle lights disabled. The
camera-free IMU mode enables an IMU and lighting without entering RTX tiled
camera startup. Perception mode enables verified RGB/depth/IMU, post-render
appearance transforms, and lighting for eight parallel environments.

The measured vectorization is `1 robot x N independent environments`, which is
the PPO scale. The world/plugin and namespace contracts are deliberately kept
separate so a future `M robots x 1 environment` multi-agent scene can be added
without changing the policy or sensor interfaces.

## Stonefish-Style Comparison

Currently reproduced or approximated:

- 6-DoF rigid-body state with gravity, explicit buoyancy, CoB restoring torque,
  axis-wise linear/quadratic translational drag, angular drag, water current,
  and a diagonal added-mass term driven by finite-difference body acceleration.
- Individual reversible thrusters with nonlinear command-to-thrust curves,
  saturation, dead-zone, response lag, wrench allocation, and velocity control.
- Explicit BlueROV mass, CoM, CoB, and uniform-box inertia contract in the robot
  YAML; the inertia values are recorded and validated even though Fabric
  cloning prevents applying a custom USD tensor at runtime.
- A bounded open-water benchmark with a flat seabed, optional asset/landmark/
  obstacle plugin hooks, and vectorized independent environments.

Not yet reproduced:

- Full coupled fluid hydrodynamics, waves, free surface, vortex effects,
  nonlinear added mass, tether/contact hydrodynamics, and DVL/sonar models.
- OpenVINS/ORB-SLAM execution or cave geometry. These are separate plugins and
  are not required by the P0 physics baseline.

For the current research question, explicit drag/current/buoyancy, controllable
sensor metadata, lighting controls, and localization health are more valuable
than expensive fluid or cinematic rendering.

## Current Sim-to-Real Gaps

| Gap | Exposed now | Main missing fidelity |
| --- | --- | --- |
| Dynamics | buoyancy/CoB/drag/current/added-mass scales and actuator-wrench mass/inertia emulation are applied per environment; YAML records the uniform-box inertia contract | runtime PhysX mass/inertia mutation and coupled hydrodynamics; custom USD CoM/inertia is blocked by the Fabric-compatible scene path |
| Actuator | strength/asymmetry/dead-zone/latency/lag/noise/saturation ranges are sampled and command perturbations are applied | hardware-calibrated PWM curve and thermal/battery effects |
| Sensor | IMU/depth/camera noise, bias, timestamp jitter, frame drops, camera/IMU timestamp offset, yaw extrinsic perturbation, exposure offset, and image-space motion-blur hooks are applied when sensors are enabled | calibrated camera/IMU noise, full 6-DoF extrinsics, renderer-native exposure and motion blur |
| Appearance | visibility, attenuation, backscatter, exposure/motion-blur post-render hooks, ambient/light controls, and parallel RGB/depth runtime | per-episode USD light-scale mutation, wavelength model, and textures |
| Localization | GT/VIO API, health fields, JSONL/UDP transport, and a degraded IMU adapter for end-to-end contract tests | actual OpenVINS/ORB-SLAM process and relocalization dynamics |
| Geometry | scale and obstacle-density ranges, asset plugin | reconstructed cave/site assets and semantic landmarks |

Next priority is connecting a real VIO process through the existing UDP/JSONL
boundary. The physics/PPO baseline should remain unchanged while those P1/P2
components are iterated.

This preserves the intended research question: measure when underwater VIO
remains geometrically reliable under randomized dynamics, sensing, lighting,
and localization failures, then compare GT, VIO, and health-aware navigation on
unseen geometry and appearance.

## Camera repair details

With NVIDIA `595.84 open`, both `scripts/smoke_perception.py` and a minimal
rendering launch exited with code 139 in `librtx.scenedb.plugin.so` during
`UsdContext::newStage`. Restricting Vulkan to the NVIDIA ICD, clearing shader
caches, changing renderers, and booting once with `intel_iommu=off` did not
change the crash. Switching to Ubuntu's signed NVIDIA `580.173.02 open` driver
removed the native crash while retaining the original IOMMU configuration.

The first multi-environment run then exposed a separate project issue:
`clone_in_fabric=True` created only one discoverable USD camera prim. Perception
now uses USD environment cloning, while physics and camera-free IMU modes keep
Fabric cloning. Vehicle lights are spawned before USD cloning and after Fabric
cloning so both paths remain valid. Final checks pass for 1, 2, and 8 camera
environments, including the default 8 environments for 30 steps.

## BlueROV2 asset integration

The detailed visual comes from `bvibhav/stonefish_bluerov2` at commit
`6448383af6b7ef6083b0eac2c08102660591e318` under its Apache-2.0 license. The
vendored source and attribution are in `assets/bluerov2/`. Isaac Sim's mesh
converter produces component USD files, and
`assets/bluerov2/bluerov2_visual.usda` composes the hull, heavy frame, and eight
propellers with the upstream Stonefish placements.

This is intentionally a visual-only import. The detailed mesh is disabled for
collision and the existing project-owned primitive remains the rigid body used
for low-VRAM vectorized physics, mass, buoyancy, drag, and PPO. The dedicated
check `scripts/smoke_robot_visual.py` passes with 10 visual meshes, one
primitive collision, and no collision API below `/Robot/Visual`. The 8-env RGB /
depth regression also passes after the import:
`logs/perception_bluerov2_envs8_steps30.log`.
