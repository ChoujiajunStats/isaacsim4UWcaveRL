# Cave Asset Pipeline

Large cave meshes remain outside this repository. Set the external root before
running conversion and validation:

```bash
export ISAAC_UNDERWATER_ASSET_ROOT=/home/kenton/Downloads/assets
```

The first supported real cave is `porth_yr_ogof_sump9`. Its visual OBJ and
independent collision OBJ are converted into ignored files under
`outputs/cave_assets/porth_yr_ogof_sump9/`:

```bash
./.venv/bin/python scripts/convert_cave_asset.py \
  --config worlds/porth_yr_ogof_sump9.yaml --headless --device cpu
./.venv/bin/python scripts/smoke_cave_asset.py \
  --config worlds/porth_yr_ogof_sump9.yaml --num_envs 1 --steps 2 \
  --headless --device cuda:0
./.venv/bin/python scripts/check_cave_centerline.py \
  --config configs/worlds/porth_yr_ogof_sump9.yaml
PYTHONPATH=source/isaac_underwater ./.venv/bin/python \
  scripts/check_cave_portals.py --config worlds/porth_yr_ogof_sump9.yaml
```

The smoke test verifies the USD stage contract: a visual mesh is loaded with
no visual collision API, a separate collision mesh has a triangle collision
API, and every cloned environment contains both prims. It does not prove that
the mesh is watertight or that a full route is collision free.

## Current Results

| Check | Result | Evidence |
| --- | --- | --- |
| synthetic visual/collision USD conversion | PASS | `outputs/cave_assets/synthetic_cave_turn90/manifest.json` |
| synthetic 2-env stage clone | PASS | `scripts/smoke_cave_asset.py` |
| Porth visual/collision USD conversion | PASS | `outputs/cave_assets/porth_yr_ogof_sump9/manifest.json` |
| Porth 1-env GPU visual/collision smoke | PASS | `scripts/smoke_cave_asset.py` |
| Porth 1-env stereo RGB/depth/lights smoke | PASS | `scripts/smoke_cave_stereo.py` |
| Porth 2-env stereo cave clone | PASS | `scripts/smoke_cave_stereo.py` |
| Porth spawn clearance at chainage 2 m | PASS, 0.727 m | `logs/cave_clearance/porth_yr_ogof_sump9.json` |
| Porth goal clearance at chainage 20 m | PASS, 0.684 m | `logs/cave_clearance/porth_yr_ogof_sump9.json` |
| automatic Porth portal inference | PASS, chainage 4.00 m; exterior run 4.00 m; clearance 0.805 m | `logs/cave_portals/porth_yr_ogof_sump9.json` |
| outside-to-inside portal event and PPO wiring | PASS | `scripts/smoke_cave_entry.py`, `scripts/run_cave_entry_rl_pipeline.sh` |
| Full centerline clearance | NOT YET VALIDATED | minimum observed 0.183 m; mesh is not watertight |
| collision-only cave direct PPO scene | NOT YET VALIDATED | no-camera DirectRLEnv path was stopped after prolonged PhysX initialization |

Legacy spawn and goal positions are derived from the supplied provisional
centerline. The entry curriculum additionally infers a portal from the
automatic skeleton's endpoint surface-clearance transition; it does not read a
manually painted entrance coordinate. Automatic inference on arbitrary open
photogrammetry meshes remains a held-out validation task.

The cave task registration is:

```text
Isaac-Underwater-Cave-Perception-v0   visual + RGB/depth/IMU scene smoke
Isaac-Underwater-Cave-Stereo-v0       visual + stereo RGB/depth/IMU/lights
Isaac-Underwater-Cave-VisualPilot-v0  visual actor + privileged critic PPO
Isaac-Underwater-Cave-Explore-v0      no-goal stereo CNN + GRU exploration PPO
Isaac-Underwater-Cave-Entry-v0        automatic-portal vision-only entry PPO
```

The stereo cave task now resolves the default Porth world directly through the
registered Gym task and has been validated at one and two environments. The
raw cave tree remains external and is resolved through
`ISAAC_UNDERWATER_ASSET_ROOT`.

The visual pilot now consumes a fixed 1553-D stereo RGB/depth/IMU/pressure
observation with a 19-D privileged navigation critic and has passed finite
train/evaluate/checkpoint-load smoke. Cave route progress and centerline
deviation are logged as provisional teacher signals. A PhysX contact reporter
is available, but GPU PhysX 5.1 cannot filter the imported triangle collider
against a cave-only path; its force is therefore a conservative static
collider proxy, not validated cave-only collision ground truth. Strict contact
termination is disabled by default until this limitation is resolved.

The exploration task does not expose the provisional centerline, goal, or an
entrance coordinate to its actor. Its reward uses a per-environment workspace
visitation grid and depth clearance. It uses one fixed, collision-checked,
centerline-derived spawn and has no entry event, so its coverage score is not
proof of automatic entrance discovery.

The entry task starts 1.5--2.5 m outside the inferred Porth portal, randomizes
yaw, and terminates successfully after a sparse portal-crossing event. Its
1549-D actor still receives no goal, centerline, portal coordinate, or
visitation grid. A one-update train/evaluate/checkpoint gate passes; policy
convergence and cross-cave portal inference are not yet validated.
