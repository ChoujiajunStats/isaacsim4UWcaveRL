"""Low-cost BlueROV2 rigid body and optional detailed visual."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


def cuboid_inertia_kg_m2(mass_kg: float, size_m: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return principal inertia for a uniform cuboid about its center of mass."""
    x, y, z = size_m
    factor = mass_kg / 12.0
    return (factor * (y * y + z * z), factor * (x * x + z * z), factor * (x * x + y * y))


def bluerov2_visual_usd_path() -> Path:
    """Return the project-owned path to the converted BlueROV2 visual asset."""
    return Path(__file__).resolve().parents[4] / "assets" / "bluerov2" / "bluerov2_visual.usda"


def spawn_bluerov2_visual(robot_prim_path: str) -> None:
    """Attach the detailed visual while retaining the primitive collision body."""
    from pxr import UsdGeom

    usd_path = bluerov2_visual_usd_path()
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"BlueROV2 visual USD is missing: {usd_path}. "
            "Run scripts/convert_bluerov2_assets.py to regenerate it."
        )

    stage = sim_utils.get_current_stage()
    collision_visual = stage.GetPrimAtPath(f"{robot_prim_path}/geometry/mesh")
    if not collision_visual.IsValid():
        raise RuntimeError(f"Primitive robot geometry is missing below {robot_prim_path}")
    UsdGeom.Imageable(collision_visual).MakeInvisible()

    visual_cfg = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
        collision_props=None,
        rigid_props=None,
        mass_props=None,
    )
    visual_cfg.func(f"{robot_prim_path}/Visual", visual_cfg)


def make_underwater_robot_cfg(
    robot_cfg: dict,
    *,
    activate_contact_sensors: bool = False,
) -> RigidObjectCfg:
    """Build the lightweight BlueROV rigid body configuration.

    Contact reporting is opt-in because PhysX contact reporters add memory and
    bookkeeping that are unnecessary for open-water throughput benchmarks.
    Cave tasks can enable it without changing the collision geometry.
    """
    geometry = robot_cfg["geometry"]
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=tuple(geometry["size_m"]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=robot_cfg["max_linear_velocity_mps"],
            max_angular_velocity=math.degrees(robot_cfg["max_angular_velocity_radps"]),
                solver_position_iteration_count=robot_cfg["solver_position_iterations"],
                solver_velocity_iteration_count=robot_cfg["solver_velocity_iterations"],
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=robot_cfg["mass_kg"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(geometry["color_rgb"])),
            activate_contact_sensors=activate_contact_sensors,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 4.0)),
    )
