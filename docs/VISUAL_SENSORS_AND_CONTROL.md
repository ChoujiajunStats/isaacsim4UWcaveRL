# BlueROV2 Visual Sensors and Control Contract

## Current status

The visual task is now `Isaac-Underwater-PointNav-Stereo-v0`; the cave variant
is `Isaac-Underwater-Cave-Stereo-v0`. Both expose two
independent Isaac `TiledCamera` sensors and emits synchronized
`rgb_left/rgb_right/depth_left/depth_right` fields in `SensorPacket`. The
legacy `rgb/depth` fields alias the left camera, so existing VIO transport and
logging code remains compatible.

The candidate stereo geometry is taken from
`assets/bluerov2/source/bluerov2.scn`:

| item | value | status |
| --- | --- | --- |
| left position (body frame) | `[0.16, -0.0725, 0.15] m` | source-derived candidate |
| right position (body frame) | `[0.16, 0.0725, 0.15] m` | source-derived candidate |
| position-derived baseline | `0.145 m` | source-derived candidate |
| right-camera `baseline="-60.5"` scene attribute | unresolved | NOT YET VALIDATED |
| intrinsics | `640x480, 75 deg` in Stonefish; Isaac smoke uses low-cost `160x120` | calibration candidate |
| Isaac stereo near plane | `0.25 m` | estimated opaque-hull self-occlusion guard |

The `-60.5` Stonefish attribute is not silently treated as millimetres or as
ground truth. A real stereo calibration or measured camera mount is required
before sim-to-real claims.

The source camera `x=0.160 m` lies inside the converted opaque hull, whose
front bound is approximately `x=0.222 m`. With the original `0.05 m` near
plane, the central depth statistic read the vehicle itself (`0.159 m`). The
stereo tasks therefore use a `0.25 m` near plane; the same cave spawn then
measured `4.496 m`. This is a documented rendering guard, not a claim about
the real housing or camera extrinsics.

## Active lights

`configs/lighting/bluerov2.yaml` records the provenance. Blue Robotics
manufacturer documentation supports two or four lamps, `1500 lm` each, with a
`135 deg` beam. Exact lamp XYZ poses, optical axes, and the Isaac RTX
lumen-to-pixel calibration are **NOT YET VALIDATED**. The current symmetric
poses are therefore marked `published+estimated_pose` and remain calibration
parameters.

The stereo task uses six normalized action values:

```text
[desired_surge, desired_sway, desired_heave, desired_yaw_rate,
 light_left_scale, light_right_scale]
```

Light action `-1..1` maps to intensity scale `0..1`. The scale is stored per
environment and is applied only to USD light intensity; it does not add a
force or alter hydrodynamics. The API
`set_light_intensity_scale([num_envs, num_lights])` is available for scripted
calibration and future illumination policies. The original four-action PPO
task is unchanged.

## 6-DoF control decision

The allocator already spans all six body wrench axes. `WrenchController` now
accepts either the legacy four-dimensional `Fx,Fy,Fz,Tz` action or a full
`Fx,Fy,Fz,Tx,Ty,Tz` action. This is a control-contract smoke capability, not a
claim that a visual policy should directly output pose.

For cave navigation the recommended hierarchy remains:

```text
RGB/stereo/depth policy -> local velocity + yaw-rate (or waypoint)
                       -> 6-DoF inner wrench controller
                       -> 8-thruster allocator
                       -> hydrodynamics + PhysX
```

Direct pose actions are harder to keep feasible near cave walls and make
partial-observation credit assignment worse. First validate the inner wrench
loop with step and coast-down tests, then expose six-wrench actions only for a
dedicated low-level benchmark.

## MuJoCo recommendation

MuJoCo is optional and should not replace Isaac Sim for the final visual cave
task. It is useful for fast low-level controller ablations or pretraining if
the Isaac inner-loop response is demonstrably unstable or too slow to tune.
It will not solve stereo calibration, underwater appearance, active-light
photometry, cave geometry, or Isaac/real hydrodynamic mismatch. If introduced,
the MuJoCo model must use the same body-frame signs, six-wrench action
semantics, thruster layout, actuator lag, and observation units, and the policy
must be replayed and validated in Isaac afterwards.

## Validation gate

Run `scripts/smoke_stereo_lighting.py` with cameras enabled for open water,
and `scripts/smoke_cave_stereo.py` for the Porth cave. The gate is
`PASS` only when both images are non-null, the packet baseline metadata is
`0.145 m`, and per-environment lamp intensity changes are visible in USD.
Photometric realism and real stereo depth accuracy remain **NOT YET
VALIDATED** until a camera/lighting calibration is supplied.

Latest local run (RTX 4060, Isaac Sim 5.1, 2 environments, 2 environment
steps) completed:

```text
stereo_lighting_smoke: PASS
RGB: (120, 160, 3) per camera
position-derived baseline: 0.145 m
per-environment opposite lamp schedules: PASS
```

This validates sensor/command plumbing and USD light independence. It does
not validate real-world stereo depth accuracy, lumen calibration, or lamp
mount geometry.

The Porth cave stereo smoke also passed locally with one and two environments
and two steps:

```text
cave_stereo_smoke: PASS
visual_meshes=1 collision_meshes=1 baseline_m=0.145
```

This confirms that the cave visual and collision prims coexist with both
cameras and active lights. It does not certify the full cave route as
collision-free; the existing centerline clearance report remains the
authoritative gate for that question.
