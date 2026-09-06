# Underwater Dynamics Validation

Validation date: 2026-09-06. The vehicle represented by the detailed visual
asset has four horizontal and four vertical thrusters, so it is treated as an
8-thruster Heavy-like configuration. Standard 6-thruster BlueROV2 values are
not silently used as ground truth.

## 1. Implemented equations

The documented marine-craft structure is

```text
M nu_dot + C_RB(nu) nu + C_A(nu_r) nu_r
    + D(nu_r) nu_r + g_hydro(eta) = tau_external
```

with

```text
eta = [x, y, z, roll, pitch, yaw]
nu = [u, v, w, p, q, r]
nu_r = [R(q)^T (v_vehicle_world - v_water_world), p, q, r]
M = M_RB + M_A
D(nu_r) = diag(d_linear + d_quadratic * abs(nu_r))
```

The runtime layer computes buoyancy/restoring wrench, relative-current linear
and quadratic damping, optional diagonal/full added-mass acceleration and
energy-neutral added-mass Coriolis terms, and actuator wrench. The body frame
is x-forward, y-left, z-up and quaternions are wxyz.

`M_RB`, `M_A`, `C_RB`, `C_A`, and `D` are exposed by
`isaac_underwater.physics.MarineDynamics`. The compatibility class
`Hydrodynamics` uses the same implementation and returns external force and
torque tensors for a batch of environments.

## 2. PhysX responsibility and double-counting boundary

PhysX owns the rigid-body mass, rigid-body inertia, gravity, pose/velocity
integration, gyroscopic rigid-body dynamics, collision, and contacts. The
underwater layer does **not** integrate a second rigid body and does not add a
second weight force. It adds upward buoyancy at CoB, CoB restoring moment,
hydrodynamic damping based on relative water velocity, optional added-mass
correction, current, and thruster forces/moments through
`instantaneous_wrench_composer`.

The standalone response harness in `scripts/run_hydro_validation.py` integrates
the same equation only to validate wrench terms and timestep behavior. It is
not used by the Isaac task.

## 3. Presets

| Preset | Runtime terms | Intended use | Evidence status |
| --- | --- | --- | --- |
| `fast_rl` | buoyancy/restoring, linear + quadratic damping, current, 8 thrusters, lag | high parallel PPO | structure verified; coefficients mostly estimated |
| `hydro_rl` | `fast_rl` plus diagonal `M_A` and optional `C_A` | primary training model | added mass is NOT YET VALIDATED |
| `reference` | full-matrix interface, diagnostics, logging | low-env validation/replay | no full identified matrix yet |

The PointNav task defaults to `fast_rl`; `scripts/smoke_pointnav.py` accepts
`--hydrodynamics_preset hydro_rl` or `reference` and all presets use the same
BlueROV body, controller, thruster layout, and visual/physics separation.

## 4. Parameter provenance

| Parameter | Nominal | Unit | Source type | Confidence / status |
| --- | ---: | --- | --- | --- |
| mass | 20.0 | kg | estimated project rigid-body contract | low; weigh real payload |
| inertia diagonal | [0.5667, 1.2167, 1.4833] | kg m^2 | lightweight cuboid geometry | low; detailed mesh/payload not represented |
| displaced volume | 0.0197073 | m^3 | estimated project hydrostatic nominal | low; ballast state not measured |
| CoM | [0, 0, 0] | m | project origin convention | low |
| CoB | [0, 0, 0.04] | m | estimated project nominal | low; not measured |
| linear/quadratic drag | see YAML | SI axis coefficients | estimated project nominal | low; calibration required |
| added mass | [2,3,4] kg and [0.05,0.08,0.12] kg m^2 | SI | conservative approximation | NOT YET VALIDATED |
| thruster max/lag | 40 N forward, 30 N reverse, 0.08 s | N, s | project actuator contract | low; PWM/current log required |
| heave alpha | 2.81 | 1/m | Espinal et al. (2026), Table 5 | experimental single-axis reference only |

Full ranges, source URLs, vehicle version, notes, and confidence are in
`configs/robots/bluerov2_hydro.yaml`.

`isaac_underwater.calibration` now exposes provenance-aware parameter records
and a weighted trajectory loss. `scripts/fit_hydro_params.py` validates a
unified trajectory but deliberately refuses to fit parameters until a
reference/real trajectory has passed frame, unit, and same-command replay
checks.

## 5. Analytical verification

Command: `.venv/bin/python scripts/validate_physics_invariants.py`.

Result: `PASS` for 10,000 random states. The generated evidence is
`benchmarks/analytical_invariants.json`.

| Check | Result |
| --- | --- |
| mass matrix symmetry | PASS, max error 0 |
| mass matrix positive definiteness | PASS, eigenvalues [0.6167, 1.2967, 1.6033, 22, 23, 24] |
| Coriolis energy property | PASS, max `abs(nu^T C nu)` = 4.34e-5 (float32 tolerance) |
| damping matrix dissipation | PASS, minimum sampled power 0.7851 |
| damping wrench dissipation | PASS |
| finite / NaN / Inf | PASS |
| zero relative current drag | PASS |
| stationary-current force direction | PASS |
| roll and pitch restoring direction | PASS |

## 6. Physical response tests

Command: `.venv/bin/python scripts/run_hydro_validation.py`.

All P0 tests pass in the standalone wrench harness at `dt=1/120`:

| Test | Result |
| --- | --- |
| T0 neutral release | PASS |
| T1 positive buoyancy rises | PASS |
| T2 roll restoring | PASS |
| T3 pitch restoring | PASS |
| T4 surge coast-down | PASS |
| T5 sway coast-down | PASS |
| T6 heave coast-down | PASS |
| T7 yaw decay | PASS |
| T8 current relative-velocity invariant | PASS |
| T9 all 8 thruster directions | PASS |

Unified trajectory output is `benchmarks/hydro_trajectories.csv`; plots are in
`benchmarks/plots/` (`surge_step.png`, `surge_coastdown.png`,
`heave_response.png`, `yaw_response.png`, `restoring_roll.png`,
`current_response.png`).

## 7. Numerical convergence

Command: `.venv/bin/python scripts/timestep_convergence.py`.

The same combined surge/yaw wrench was run at all three steps against the
`dt=1/240` trajectory:

| dt | position RMSE (m) | velocity RMSE |
| ---: | ---: | ---: |
| 1/60 | 1.2574e-3 | 5.2966e-4 |
| 1/120 | 4.1826e-4 | 1.7655e-4 |
| 1/240 | 0 | 0 |

The errors decrease monotonically as dt decreases. Raw data is
`benchmarks/timestep_convergence.csv`, with plot
`benchmarks/plots/timestep_convergence.png`.

## 8. RTX 4060 scaling benchmark

Command: `.venv/bin/python scripts/benchmark_hydrodynamics_scaling.py
--presets fast_rl hydro_rl --envs 32 64 128 256 512 --warmup_steps 20
--measure_steps 100`.

Measured on the RTX 4060 Laptop GPU (driver 580.173.02). VRAM is board usage
sampled after warmup, not a guarantee for a desktop session with other GPU
workloads.

| Preset | Envs | VRAM MiB | Env steps/s | Sim steps/s | RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast_rl | 32 | 1573 | 3474.59 | 325.74 | 5.429 |
| fast_rl | 64 | 1551 | 6574.65 | 308.19 | 5.136 |
| fast_rl | 128 | 1536 | 13519.47 | 316.86 | 5.281 |
| fast_rl | 256 | 1527 | 25901.86 | 303.54 | 5.059 |
| fast_rl | 512 | 1553 | 51314.91 | 300.67 | 5.011 |
| hydro_rl | 32 | 1587 | 2761.86 | 258.92 | 4.315 |
| hydro_rl | 64 | 1582 | 5611.46 | 263.04 | 4.384 |
| hydro_rl | 128 | 1585 | 11896.71 | 278.83 | 4.647 |
| hydro_rl | 256 | 1586 | 24073.66 | 282.11 | 4.702 |
| hydro_rl | 512 | 1594 | 48619.13 | 284.88 | 4.748 |
| reference | 32 | 1561 | 3061.63 | 287.03 | 4.784 |

The added-mass path costs throughput but remains below 1.6 GiB in this
physics-only scene. The benchmark does not validate parameters; it only
measures the vectorized implementation and stability envelope. Raw files are
`logs/benchmarks/hydrodynamics_fast_rl.csv`,
`logs/benchmarks/hydrodynamics_hydro_rl.csv`, and
`logs/benchmarks/hydrodynamics_reference.csv`.

## 9. Reference-model and Stonefish status

`docs/reference_models.md` records Fossen and the experimentally identified
BlueROV2 Heavy heave reference. Same-command Isaac versus that experimental
trajectory replay is **NOT YET VALIDATED**: the paper publishes a single-axis
closed-loop model, not a six-axis trajectory dataset or full parameter matrix.

The Stonefish visual repository and commit are recorded, but Stonefish runtime
is not installed in this workspace. `scripts/compare_stonefish.py` provides a
strict unified trajectory CSV importer and reports `NOT AVAILABLE` without an
input file. Isaac-versus-Stonefish quantitative cross-validation is therefore
**NOT YET VALIDATED** and no equivalence claim is made.

## 10. Limitations and next calibration steps

- Current mass, inertia, CoM, CoB, damping, added mass, and thruster curve
  values are mostly estimated or geometry-derived. They are exposed for
  calibration, not presented as measured truth.
- The current Isaac body uses a lightweight primitive collision geometry and a
  detailed visual-only mesh; their dimensions and inertia are not identical.
- Added mass is diagonal in `hydro_rl`; full coupling is an interface and
  diagnostic path, not a validated matrix. The finite-difference acceleration
  correction is an external-wrench approximation.
- No CFD, particle water, waves, tether, wall, vortex, DVL, or sonar physics is
  claimed.
- Real-world validation, Stonefish replay, and six-axis parameter fitting are
  NOT YET VALIDATED. The next measurements should be independent mass/CoB/CoM,
  axis-isolated coast-down and step tests, and per-thruster PWM/current logs.

## 11. Required answers

1. **Basic marine mathematical properties?** Yes for the implemented
   structure: analytical symmetry/positive-definiteness, Coriolis energy
   neutrality, damping dissipation, finite values, and restoring/current
   invariants all PASS.
2. **Published/experimental parameters?** The governing Fossen structure and
   the BlueROV2 Heavy single-axis heave coefficients are published evidence.
   Most 6-DoF nominal values are not.
3. **Approximations?** Mass/inertia/volume/CoM/CoB, all current damping,
   diagonal added mass, and actuator curve/lag are estimated or project
   contracts; provenance says so explicitly.
4. **Isaac versus experimental benchmark?** Quantitative same-command
   reproduction is NOT YET VALIDATED; the available paper is a single-axis
   closed-loop reference, not a six-axis replay.
5. **Stonefish agreement/difference?** No quantitative comparison has been
   run. Importer is ready; Stonefish runtime is unavailable here.
6. **Enough for PPO/VIO/navigation?** The verified structure and stable
   vectorized wrench layer are adequate as an engineering baseline for PPO and
   motion-excitation trajectory generation. Fidelity is not yet sufficient to
   claim sim-to-real equivalence for navigation.
7. **Largest sim-to-real gap?** Vehicle-specific hydrodynamic coefficients and
   actuator response, especially coupled/added-mass and drag terms, followed by
   the primitive-vs-real geometry/inertia mismatch.
