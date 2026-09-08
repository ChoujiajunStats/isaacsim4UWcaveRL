"""World plug-in contract; caves remain assets rather than simulator forks."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class OpenWaterWorldCfg:
    size_m: tuple[float, float, float] = (20.0, 20.0, 8.0)
    seabed_depth_m: float = 8.0
    asset_path: str | None = None
    collision_asset_path: str | None = None
    asset_scale: float = 1.0
    asset_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    asset_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    asset_name: str = "Asset"
    clone_per_env: bool = False
    navigation_workspace_size_m: tuple[float, float, float] | None = None
    visual_enabled: bool = True
    obstacles_enabled: bool = False
    landmarks_enabled: bool = False
    landmark_positions_m: tuple[tuple[float, float, float], ...] = ()
    obstacle_positions_m: tuple[tuple[float, float, float], ...] = ()
    obstacle_size_m: tuple[float, float, float] = (1.0, 1.0, 1.0)


class WorldAssetPlugin(Protocol):
    def spawn(self, prim_path: str, cfg: OpenWaterWorldCfg) -> None:
        """Spawn a world asset under the supplied USD prim path."""


def spawn_open_water_world(
    cfg: OpenWaterWorldCfg,
    *,
    root_path: str = "/World",
    seabed_path: str | None = None,
) -> list[object]:
    """Spawn the benchmark seabed plus optional asset/landmark/obstacle prims.

    The function is intentionally a thin plugin boundary. A future cave or
    reconstructed site can replace the ``asset_path`` branch without changing
    the robot, sensor, or RL task.
    """
    from isaaclab import sim as sim_utils
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    prims: list[object] = []
    floor_path = seabed_path or f"{root_path}/seabed"
    prims.append(
        spawn_ground_plane(
            prim_path=floor_path,
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.8,
                    dynamic_friction=0.6,
                    restitution=0.0,
                )
            ),
        )
    )
    if cfg.asset_path and cfg.visual_enabled:
        asset_cfg = sim_utils.UsdFileCfg(
            usd_path=cfg.asset_path,
            scale=(cfg.asset_scale,) * 3,
        )
        prims.append(
            asset_cfg.func(
                f"{root_path}/{cfg.asset_name}",
                asset_cfg,
                translation=cfg.asset_translation_m,
                orientation=cfg.asset_orientation_wxyz,
            )
        )
    if cfg.collision_asset_path:
        collision_cfg = sim_utils.UsdFileCfg(
            usd_path=cfg.collision_asset_path,
            scale=(cfg.asset_scale,) * 3,
        )
        prims.append(
            collision_cfg.func(
                f"{root_path}/{cfg.asset_name}Collision",
                collision_cfg,
                translation=cfg.asset_translation_m,
                orientation=cfg.asset_orientation_wxyz,
            )
        )

    marker_cfg = sim_utils.CuboidCfg(
        size=(0.25, 0.25, 0.25),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.65, 0.08)),
    )
    if cfg.landmarks_enabled:
        for index, position in enumerate(cfg.landmark_positions_m):
            prims.append(marker_cfg.func(f"{root_path}/Landmarks/Marker_{index:03d}", marker_cfg, translation=position))

    if cfg.obstacles_enabled:
        obstacle_cfg = sim_utils.CuboidCfg(
            size=cfg.obstacle_size_m,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.22, 0.24)),
        )
        for index, position in enumerate(cfg.obstacle_positions_m):
            prims.append(obstacle_cfg.func(f"{root_path}/Obstacles/Obstacle_{index:03d}", obstacle_cfg, translation=position))
    return prims
