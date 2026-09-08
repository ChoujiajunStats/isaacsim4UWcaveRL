"""Low-VRAM vectorized underwater point-navigation environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, Imu, ImuCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaac_underwater.actuators import ThrusterAllocator, ThrusterModel, bluerov2_thruster_layout
from isaac_underwater.appearance import (
    UnderwaterAppearanceCfg,
    UnderwaterLightingCfg,
    VehicleLightCfg,
    apply_underwater_appearance,
    bluerov2_candidate_lighting,
    coerce_lighting_cfg,
    spawn_underwater_lighting,
    update_underwater_lighting_batch,
)
from isaac_underwater.config import load_config
from isaac_underwater.controllers import VelocityController, command_to_thruster
from isaac_underwater.interfaces import ControlCommand, GroundTruthState, LocalizationOutput, SensorPacket
from isaac_underwater.localization import (
    ExternalVIOBackend,
    GroundTruthLocalizationBackend,
)
from isaac_underwater.logging import EpisodeJsonlLogger
from isaac_underwater.navigation import (
    CavePortal,
    PolicyStateSource,
    VoxelVisitTracker,
    balanced_scene_assignment,
    build_navigation_observation,
    build_visual_observation,
    infer_clearance_portals,
    interpolate_polyline,
    load_cave_route,
    navigation_observation_to_tensor,
    robust_forward_clearance,
    visual_feature_dim,
)
from isaac_underwater.physics import (
    CurrentField,
    CurrentProfileCfg,
    Hydrodynamics,
    HydrodynamicsCfg,
    rotate_body_to_world,
    rotate_world_to_body,
)
from isaac_underwater.randomization import (
    domain_gap_from_mapping,
    perturb_thruster_command,
    rotate_body_vectors_z,
    sample_domain_gap,
)
from isaac_underwater.robot import make_underwater_robot_cfg, spawn_bluerov2_visual
from isaac_underwater.sensors import SensorSuiteCfg
from isaac_underwater.worlds import (
    CaveWorldCfg,
    OpenWaterWorldCfg,
    coerce_cave_scene_cfg,
    load_cave_dataset_world_cfg,
    load_cave_world_cfg,
    spawn_open_water_world,
    spawn_world_assets,
)


@configclass
class UnderwaterPointNavEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 3
    action_space = 4
    observation_space = 17
    state_space = 0
    debug_vis = False

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=decimation,
        device="cuda:0",
        use_fabric=True,
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**18,
            gpu_max_rigid_patch_count=2**14,
            gpu_found_lost_pairs_capacity=2**16,
            gpu_found_lost_aggregate_pairs_capacity=2**16,
            gpu_total_aggregate_pairs_capacity=2**16,
            gpu_heap_capacity=2**24,
            gpu_temp_buffer_capacity=2**23,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=24.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    robot: RigidObjectCfg = make_underwater_robot_cfg(load_config("robot.yaml"))
    camera_sensor: TiledCameraCfg | None = None
    camera_left_sensor: TiledCameraCfg | None = None
    camera_right_sensor: TiledCameraCfg | None = None
    imu_sensor: ImuCfg | None = None
    contact_sensor: ContactSensorCfg | None = None
    lighting: UnderwaterLightingCfg = UnderwaterLightingCfg()
    appearance: UnderwaterAppearanceCfg = UnderwaterAppearanceCfg()
    world: OpenWaterWorldCfg = OpenWaterWorldCfg()

    workspace_size_m = (20.0, 20.0, 8.0)
    reset_xy_m = 4.0
    reset_depth_range_m = (2.0, 6.0)
    goal_distance_range_m = (3.0, 8.0)
    goal_vertical_offset_m = 2.0
    goal_threshold_m = 0.5
    max_command_velocity_mps = (1.2, 1.0, 0.8)
    max_command_yaw_rate_radps = 1.0
    policy_state_source = "ground_truth"
    current_mode = "constant"
    current_velocity_w_mps = (0.0, 0.0, 0.0)
    current_amplitude_w_mps = (0.0, 0.0, 0.0)
    current_period_s = 60.0
    current_random_walk_std_mps = 0.01
    current_max_speed_mps = 0.5
    hydrodynamics_preset = "fast_rl"
    episode_log_path: str | None = None
    domain_randomization_enabled = False
    detailed_robot_visual = True
    light_control_enabled = False
    light_control_channels = 0
    light_control_default_scale = 1.0
    visual_observation_enabled = False
    visual_output_hw = (12, 16)
    visual_max_depth_m = 12.0
    visual_mission_command_enabled = True

    # Label-free exploration is a separate task contract.  The visitation
    # grid and GT pose are privileged reward/critic inputs and are never
    # included in the deployed actor observation.
    exploration_reward_enabled = False
    exploration_voxel_size_m = 0.5
    exploration_new_voxel_reward = 1.0
    exploration_revisit_penalty = 0.002
    exploration_clearance_margin_m = 0.8
    exploration_clearance_penalty_weight = 2.0
    exploration_depth_crop_fraction = 0.5
    exploration_depth_quantile = 0.1
    exploration_random_yaw = False

    # Optional geometry-derived entry curriculum.  Portal candidates come
    # from an automatic surface-clearance transition at a skeleton endpoint,
    # never from a hand-authored entrance coordinate or actor command.
    cave_entry_enabled = False
    cave_entry_minimum_clearance_m = 0.65
    cave_entry_minimum_exterior_run_m = 1.0
    cave_entry_tangent_probe_m = 1.5
    cave_entry_spawn_distance_range_m = (1.5, 2.5)
    cave_entry_spawn_lateral_jitter_m = 0.10
    cave_entry_spawn_vertical_jitter_m = 0.05
    cave_entry_gate_radius_m = 1.5
    cave_entry_depth_m = 0.75
    cave_entry_bonus = 25.0
    cave_entry_terminate_on_success = True

    # Cave route/contact terms are inactive for open-water tasks.  They are
    # explicit configuration so their provisional status is visible in logs.
    cave_route_reward_enabled = False
    route_progress_weight = 2.0
    route_deviation_penalty_weight = 0.5
    route_deviation_soft_m = 0.75
    route_deviation_hard_m = 3.0
    centerline_clearance_penalty_weight = 1.0
    vehicle_bounding_radius_m = 0.5
    clearance_margin_m = 0.05
    collision_force_threshold_n = 5.0
    collision_penalty = 20.0
    cave_contact_termination_enabled = False
    cave_spawn_yaw_jitter_rad = 0.0

    position_scale = 0.1
    linear_velocity_scale = 0.5
    angular_velocity_scale = 0.5
    progress_weight = 10.0
    goal_bonus = 20.0
    heading_weight = 0.02
    action_penalty_weight = 0.005
    out_of_bounds_penalty = 10.0


class UnderwaterPointNavEnv(DirectRLEnv):
    cfg: UnderwaterPointNavEnvCfg

    def __init__(self, cfg: UnderwaterPointNavEnvCfg, render_mode: str | None = None, **kwargs):
        # Hydra recursively updates plain dataclass tuples as mappings.  Coerce
        # that representation before Isaac scene setup consumes the light list.
        cfg.lighting = coerce_lighting_cfg(cfg.lighting)
        self._camera: TiledCamera | None = None
        self._camera_left: TiledCamera | None = None
        self._camera_right: TiledCamera | None = None
        self._imu: Imu | None = None
        self._contact_sensor: ContactSensor | None = None
        self._light_prims_by_env: list[list[object]] = []
        self._last_control_command: ControlCommand | None = None
        self._last_thruster_command: torch.Tensor | None = None
        self._last_localization: LocalizationOutput | None = None
        self._held_sensor_frames: dict[str, torch.Tensor] = {}
        self._cave_scene_variants = tuple(
            coerce_cave_scene_cfg(scene)
            for scene in getattr(cfg.world, "scene_variants", ())
        )
        if self._cave_scene_variants:
            cfg.world.scene_variants = self._cave_scene_variants
        self._cave_scene_assignment_cpu: tuple[int, ...] = ()
        if self._cave_scene_variants:
            self._cave_scene_assignment_cpu = balanced_scene_assignment(
                int(cfg.scene.num_envs), len(self._cave_scene_variants)
            )
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._cave_scene_ids = torch.tensor(
            self._cave_scene_assignment_cpu or (0,) * self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        light_channels = int(getattr(cfg, "light_control_channels", 0))
        if bool(getattr(cfg, "light_control_enabled", False)) and light_channels <= 0:
            light_channels = len(cfg.lighting.vehicle_lights)
        self._light_intensity_scale = torch.full(
            (self.num_envs, light_channels),
            float(getattr(cfg, "light_control_default_scale", 1.0)),
            device=self.device,
        )
        self._desired_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._out_of_bounds = torch.zeros_like(self._success)
        self._cave_collision = torch.zeros_like(self._success)
        self._episode_had_collision = torch.zeros_like(self._success)
        self._cave_contact_force_n = torch.zeros(self.num_envs, device=self.device)
        # Keep terminal causes across DirectRLEnv's immediate reset.  The
        # policy runner only needs ``log`` scalars, while finite evaluators
        # need per-environment episode outcomes after ``step`` returns.
        self._last_episode_success = torch.zeros_like(self._success)
        self._last_episode_out_of_bounds = torch.zeros_like(self._success)
        self._last_episode_timeout = torch.zeros_like(self._success)
        self._last_episode_collision = torch.zeros_like(self._success)
        self._last_episode_workspace_coverage = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_unique_voxels = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._last_episode_path_length_m = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_reference_path_m = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_spl = torch.zeros(self.num_envs, device=self.device)
        self._episode_path_length_m = torch.zeros(self.num_envs, device=self.device)
        self._previous_episode_position_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._exploration_forward_clearance_m = torch.full(
            (self.num_envs,), float(cfg.visual_max_depth_m), device=self.device
        )
        self._cave_entry_portal_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cave_entry_signed_depth_m = torch.zeros(self.num_envs, device=self.device)
        self._cave_entry_radial_distance_m = torch.zeros(self.num_envs, device=self.device)
        self._cave_entry_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_entered_cave = torch.zeros_like(self._cave_entry_event)
        self._max_command_velocity = torch.tensor(cfg.max_command_velocity_mps, device=self.device)
        self._localization_backend = (
            ExternalVIOBackend()
            if cfg.policy_state_source == PolicyStateSource.VIO.value
            else GroundTruthLocalizationBackend()
        )
        self._episode_logger = EpisodeJsonlLogger(cfg.episode_log_path) if cfg.episode_log_path else None
        self._previous_linear_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_angular_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._domain_gap_cfg = None
        self._domain_gap_samples: dict[str, torch.Tensor] | None = None
        self._domain_command_state: torch.Tensor | None = None
        if cfg.domain_randomization_enabled:
            self._domain_gap_cfg = domain_gap_from_mapping(load_config("rl/domain_gap.yaml"))
            self._domain_gap_samples = sample_domain_gap(self._domain_gap_cfg, self.num_envs, device=self.device)
            self._domain_command_state = torch.zeros(self.num_envs, 8, device=self.device)

        water = load_config("underwater.yaml")
        hydro_contract = load_config("robots/bluerov2_hydro.yaml")
        preset = getattr(cfg, "hydrodynamics_preset", "fast_rl")
        if preset not in hydro_contract["presets"]:
            raise ValueError(f"Unknown hydrodynamics preset {preset!r}")
        water_cfg = water["water"]
        mass = hydro_contract["mass"]
        inertia = hydro_contract["inertia"]
        com = hydro_contract["center_of_mass"]
        cob = hydro_contract["center_of_buoyancy"]
        volume = hydro_contract["volume"]
        added_mass = hydro_contract["added_mass"]
        preset_cfg = hydro_contract["presets"][preset]
        full_matrix = added_mass.get("full_matrix_nominal")
        linear_damping = hydro_contract["linear_damping"]
        quadratic_damping = hydro_contract["quadratic_damping"]
        self._hydrodynamics = Hydrodynamics(
            HydrodynamicsCfg(
                water_density_kg_m3=hydro_contract["water"]["density"]["nominal"],
                displaced_volume_m3=volume["nominal"],
                gravity_mps2=hydro_contract["water"]["gravity"]["nominal"],
                mass_kg=mass["nominal"],
                inertia_kg_m2=tuple(inertia["nominal"]),
                center_of_mass_m=tuple(com["nominal"]),
                center_of_buoyancy_m=tuple(cob["nominal"]),
                linear_drag_coeff=tuple(linear_damping[axis]["nominal"] for axis in ("surge", "sway", "heave")),
                quadratic_drag_coeff=tuple(quadratic_damping[axis]["nominal"] for axis in ("surge", "sway", "heave")),
                angular_linear_drag_coeff=tuple(linear_damping[axis]["nominal"] for axis in ("roll", "pitch", "yaw")),
                angular_quadratic_drag_coeff=tuple(quadratic_damping[axis]["nominal"] for axis in ("roll", "pitch", "yaw")),
                added_mass_kg=tuple(added_mass["translational"]["nominal"]),
                added_inertia_kg_m2=tuple(added_mass["rotational"]["nominal"]),
                added_mass_matrix_kg=tuple(tuple(row) for row in full_matrix) if full_matrix else None,
                added_mass_mode=preset_cfg["added_mass_mode"],
                added_mass_coriolis=bool(preset_cfg["added_mass_coriolis"]),
                preset=preset,
                current_velocity_w_mps=tuple(water_cfg["current_velocity_w_mps"]),
            ),
            self.device,
        )
        self._current_field = CurrentField(
            CurrentProfileCfg(
                mode=cfg.current_mode,
                mean_velocity_w_mps=tuple(cfg.current_velocity_w_mps),
                amplitude_w_mps=tuple(cfg.current_amplitude_w_mps),
                period_s=cfg.current_period_s,
                random_walk_std_mps=cfg.current_random_walk_std_mps,
                max_speed_mps=cfg.current_max_speed_mps,
            ),
            self.num_envs,
            self.device,
        )
        robot = load_config("robot.yaml")
        self._velocity_controller = VelocityController(
            linear_gain=(35.0, 40.0, 45.0),
            yaw_gain=12.0,
            max_force_n=tuple(robot["max_force_n"]),
            max_yaw_torque_nm=robot["max_torque_nm"][2],
            device=self.device,
        )
        self._thrusters = ThrusterModel(bluerov2_thruster_layout(), self.device)
        self._allocator = ThrusterAllocator(self._thrusters)
        self._thrusters.reset(self.num_envs)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, device=self.device)
            for key in (
                "progress",
                "goal",
                "heading",
                "action",
                "bounds",
                "route_progress",
                "route_deviation",
                "clearance",
                "collision",
                "exploration",
                "exploration_clearance",
                "cave_entry",
            )
        }
        self._cave_centerline: torch.Tensor | None = None
        self._cave_chainage: torch.Tensor | None = None
        self._cave_surface_clearance: torch.Tensor | None = None
        self._multi_cave_centerlines: torch.Tensor | None = None
        self._multi_cave_chainages: torch.Tensor | None = None
        self._multi_cave_route_mask: torch.Tensor | None = None
        self._multi_cave_clearances: torch.Tensor | None = None
        self._cave_scene_spawn_points = torch.empty((0, 3), device=self.device)
        self._cave_scene_goal_points = torch.empty((0, 3), device=self.device)
        self._cave_scene_spawn_chainages = torch.empty(0, device=self.device)
        self._cave_scene_goal_chainages = torch.empty(0, device=self.device)
        self._cave_scene_spawn_tangents = torch.empty((0, 3), device=self.device)
        self._cave_scene_route_lengths = torch.empty(0, device=self.device)
        self._cave_scene_spawn_jitters = torch.empty(0, device=self.device)
        self._cave_centerline_distance = torch.zeros(self.num_envs, device=self.device)
        self._cave_route_chainage = torch.zeros(self.num_envs, device=self.device)
        self._previous_cave_route_chainage = torch.zeros(self.num_envs, device=self.device)
        self._cave_local_clearance = torch.full((self.num_envs,), float("inf"), device=self.device)
        centerline_path = getattr(self.cfg.world, "centerline_path", None)
        if self._cave_scene_variants:
            self._initialize_multi_cave_routes()
        elif centerline_path:
            route = load_cave_route(centerline_path)
            scale = float(getattr(self.cfg.world, "asset_scale", 1.0))
            self._cave_chainage = torch.tensor(
                route.chainage_m, device=self.device, dtype=torch.float32
            ) * abs(scale)
            self._cave_centerline = self._transform_cave_points(
                route.points_m,
                scale=scale,
                translation=getattr(self.cfg.world, "asset_translation_m", (0.0, 0.0, 0.0)),
                orientation=getattr(self.cfg.world, "asset_orientation_wxyz", (1.0, 0.0, 0.0, 0.0)),
            )
            if route.nominal_surface_clearance_m is not None:
                self._cave_surface_clearance = torch.tensor(
                    route.nominal_surface_clearance_m,
                    device=self.device,
                    dtype=torch.float32,
                ) * abs(scale)
        self._has_cave_route = self._cave_centerline is not None or self._multi_cave_centerlines is not None

        self._cave_portals: tuple[CavePortal, ...] = ()
        self._cave_portal_positions = torch.empty((0, 3), device=self.device)
        self._cave_portal_directions = torch.empty((0, 3), device=self.device)
        self._cave_portal_chainages = torch.empty(0, device=self.device)
        self._cave_portal_chainage_directions = torch.empty(0, device=self.device)
        if bool(getattr(self.cfg, "cave_entry_enabled", False)):
            if self._multi_cave_centerlines is not None:
                raise ValueError("The legacy cave-entry curriculum does not support multi-cave datasets")
            if self._cave_centerline is None or self._cave_chainage is None:
                raise ValueError("Cave entry curriculum requires a geometry-derived centerline")
            if self._cave_surface_clearance is None:
                raise ValueError("Cave entry curriculum requires nominal_surface_clearance_m")
            self._cave_portals = infer_clearance_portals(
                self._cave_centerline,
                self._cave_chainage,
                self._cave_surface_clearance,
                minimum_clearance_m=float(self.cfg.cave_entry_minimum_clearance_m),
                minimum_exterior_run_m=float(self.cfg.cave_entry_minimum_exterior_run_m),
                tangent_probe_m=float(self.cfg.cave_entry_tangent_probe_m),
            )
            if not self._cave_portals:
                raise ValueError(
                    "No automatic cave portal passed the exterior-run and clearance requirements"
                )
            self._cave_portal_positions = torch.stack(
                [portal.position_m for portal in self._cave_portals]
            )
            self._cave_portal_directions = torch.stack(
                [portal.inward_direction for portal in self._cave_portals]
            )
            self._cave_portal_chainages = torch.tensor(
                [portal.chainage_m for portal in self._cave_portals],
                dtype=torch.float32,
                device=self.device,
            )
            self._cave_portal_chainage_directions = torch.tensor(
                [1.0 if portal.endpoint == "start" else -1.0 for portal in self._cave_portals],
                dtype=torch.float32,
                device=self.device,
            )

        self._exploration_tracker: VoxelVisitTracker | None = None
        if bool(getattr(self.cfg, "exploration_reward_enabled", False)):
            workspace = getattr(self.cfg.world, "navigation_workspace_size_m", None) or self.cfg.workspace_size_m
            self._exploration_tracker = VoxelVisitTracker(
                self.num_envs,
                workspace,
                float(self.cfg.exploration_voxel_size_m),
                self.device,
            )

    def _transform_cave_points(
        self,
        points_m,
        *,
        scale: float,
        translation,
        orientation,
    ) -> torch.Tensor:
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Cave asset_scale must be finite and positive")
        points = torch.as_tensor(points_m, dtype=torch.float32, device=self.device) * scale
        if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
            raise ValueError(f"Expected cave points [N,3], got {tuple(points.shape)}")
        quaternion = torch.as_tensor(orientation, dtype=torch.float32, device=self.device)
        if quaternion.shape != (4,) or not torch.isfinite(quaternion).all():
            raise ValueError("Cave asset_orientation_wxyz must be a finite quaternion")
        quaternion = quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1.0e-8)
        transformed = rotate_body_to_world(quaternion.unsqueeze(0), points)
        offset = torch.as_tensor(translation, dtype=torch.float32, device=self.device)
        if offset.shape != (3,) or not torch.isfinite(offset).all():
            raise ValueError("Cave asset_translation_m must contain three finite values")
        return transformed + offset

    def _initialize_multi_cave_routes(self) -> None:
        """Pad heterogeneous routes for one batched nearest-route query."""
        routes = [load_cave_route(scene.navigation_path) for scene in self._cave_scene_variants]
        max_points = max(len(route.points_m) for route in routes)
        scene_count = len(routes)
        centerlines = torch.zeros((scene_count, max_points, 3), device=self.device)
        chainages = torch.zeros((scene_count, max_points), device=self.device)
        route_mask = torch.zeros((scene_count, max_points), dtype=torch.bool, device=self.device)
        clearances = torch.full((scene_count, max_points), float("inf"), device=self.device)
        spawn_points = torch.zeros((scene_count, 3), device=self.device)
        goal_points = torch.zeros((scene_count, 3), device=self.device)
        spawn_chainages = torch.zeros(scene_count, device=self.device)
        goal_chainages = torch.zeros(scene_count, device=self.device)
        spawn_tangents = torch.zeros((scene_count, 3), device=self.device)

        for scene_id, (scene, route) in enumerate(zip(self._cave_scene_variants, routes)):
            scale = float(scene.asset_scale)
            points = self._transform_cave_points(
                route.points_m,
                scale=scale,
                translation=scene.asset_translation_m,
                orientation=scene.asset_orientation_wxyz,
            )
            chainage = torch.tensor(route.chainage_m, dtype=torch.float32, device=self.device) * abs(scale)
            count = points.shape[0]
            centerlines[scene_id, :count] = points
            chainages[scene_id, :count] = chainage
            route_mask[scene_id, :count] = True
            if route.nominal_surface_clearance_m is not None:
                clearances[scene_id, :count] = torch.tensor(
                    route.nominal_surface_clearance_m,
                    dtype=torch.float32,
                    device=self.device,
                ) * abs(scale)

            if scene.spawn_chainage_m is None:
                spawn = self._transform_cave_points(
                    (route.start_m,),
                    scale=scale,
                    translation=scene.asset_translation_m,
                    orientation=scene.asset_orientation_wxyz,
                )[0]
                spawn_index = torch.argmin(torch.linalg.vector_norm(points - spawn, dim=-1))
                spawn_chainage = chainage[spawn_index]
            else:
                spawn_chainage = torch.tensor(
                    float(scene.spawn_chainage_m) * abs(scale), device=self.device
                ).clamp(chainage[0], chainage[-1])
                spawn = interpolate_polyline(points, chainage, spawn_chainage.unsqueeze(0))[0]

            if scene.goal_chainage_m is None:
                goal = self._transform_cave_points(
                    (route.goal_m,),
                    scale=scale,
                    translation=scene.asset_translation_m,
                    orientation=scene.asset_orientation_wxyz,
                )[0]
                goal_index = torch.argmin(torch.linalg.vector_norm(points - goal, dim=-1))
                goal_chainage = chainage[goal_index]
            else:
                goal_chainage = torch.tensor(
                    float(scene.goal_chainage_m) * abs(scale), device=self.device
                ).clamp(chainage[0], chainage[-1])
                goal = interpolate_polyline(points, chainage, goal_chainage.unsqueeze(0))[0]
            if goal_chainage <= spawn_chainage:
                raise ValueError(
                    f"Scene {scene.key!r} goal must lie after its entrance on the navigation route"
                )

            nearest_spawn = int(torch.argmin((chainage - spawn_chainage).abs()).item())
            next_index = min(nearest_spawn + 1, count - 1)
            if next_index == nearest_spawn:
                next_index = max(0, nearest_spawn - 1)
            tangent = points[next_index] - points[nearest_spawn]
            tangent = tangent / torch.linalg.vector_norm(tangent).clamp_min(1.0e-8)
            spawn_points[scene_id] = spawn
            goal_points[scene_id] = goal
            spawn_chainages[scene_id] = spawn_chainage
            goal_chainages[scene_id] = goal_chainage
            spawn_tangents[scene_id] = tangent

        self._multi_cave_centerlines = centerlines
        self._multi_cave_chainages = chainages
        self._multi_cave_route_mask = route_mask
        self._multi_cave_clearances = clearances
        self._cave_scene_spawn_points = spawn_points
        self._cave_scene_goal_points = goal_points
        self._cave_scene_spawn_chainages = spawn_chainages
        self._cave_scene_goal_chainages = goal_chainages
        self._cave_scene_spawn_tangents = spawn_tangents
        self._cave_scene_route_lengths = goal_chainages - spawn_chainages
        self._cave_scene_spawn_jitters = torch.tensor(
            [float(scene.spawn_jitter_m) for scene in self._cave_scene_variants],
            dtype=torch.float32,
            device=self.device,
        )

    def submit_localization_output(self, output: LocalizationOutput, namespace: str | None = None) -> None:
        """Inject an external VIO estimate when the policy state source is VIO."""
        if not isinstance(self._localization_backend, ExternalVIOBackend):
            raise RuntimeError("submit_localization_output requires policy_state_source='vio'")
        if namespace is None:
            self._localization_backend.submit(output)
        else:
            self._localization_backend.submit_for_robot(namespace, output)

    def _setup_scene(self) -> None:
        self._robot = RigidObject(self.cfg.robot)
        if self.cfg.detailed_robot_visual and (self.sim.has_gui() or self.cfg.camera_sensor is not None):
            spawn_bluerov2_visual("/World/envs/env_0/Robot")
        if self.cfg.camera_sensor is not None:
            self._camera = TiledCamera(self.cfg.camera_sensor)
        if self.cfg.camera_left_sensor is not None:
            self._camera_left = TiledCamera(self.cfg.camera_left_sensor)
        if self.cfg.camera_right_sensor is not None:
            self._camera_right = TiledCamera(self.cfg.camera_right_sensor)
        if self.cfg.imu_sensor is not None:
            self._imu = Imu(self.cfg.imu_sensor)
        if self.cfg.contact_sensor is not None:
            self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        spawn_underwater_lighting(
            self.cfg.lighting,
            include_vehicle_lights=not self.cfg.scene.clone_in_fabric,
        )
        if self._cave_scene_variants:
            # With replicate_physics=False the scene owns independent env
            # xforms.  Clone the robot/sensors first, then compose a selected
            # cave USD under each root so no env inherits env_0's geometry.
            spawn_open_water_world(
                OpenWaterWorldCfg(seabed_enabled=False),
                root_path="/World",
                seabed_path="/World/seabed",
            )
            # Independent copies are required here.  Inheriting clones mirror
            # later additions below env_0, which would silently turn every
            # environment into the first cave.
            self.scene.clone_environments(copy_from_source=True)
            for env_index, scene_id in enumerate(self._cave_scene_assignment_cpu):
                scene = self._cave_scene_variants[scene_id]
                spawn_world_assets(
                    OpenWaterWorldCfg(
                        asset_path=scene.visual_asset_path,
                        collision_asset_path=scene.collision_asset_path,
                        asset_scale=float(scene.asset_scale),
                        asset_translation_m=tuple(scene.asset_translation_m),
                        asset_orientation_wxyz=tuple(scene.asset_orientation_wxyz),
                        asset_name="Cave",
                        visual_enabled=bool(self.cfg.world.visual_enabled),
                    ),
                    root_path=f"/World/envs/env_{env_index}",
                )
        else:
            world_root = "/World/envs/env_0" if self.cfg.world.clone_per_env else "/World"
            spawn_open_water_world(self.cfg.world, root_path=world_root, seabed_path="/World/seabed")
            self.scene.clone_environments(copy_from_source=False)
        if self.cfg.lighting.enabled and self.cfg.scene.clone_in_fabric:
            for env_index in range(self.num_envs):
                spawn_underwater_lighting(
                    self.cfg.lighting,
                    root_path=f"/World/envs/env_{env_index}",
                    robot_path=f"/World/envs/env_{env_index}/Robot",
                    include_ambient=False,
                )
        stage = sim_utils.get_current_stage()
        self._light_prims_by_env = []
        for env_index in range(self.num_envs):
            env_prims = []
            for light in self.cfg.lighting.vehicle_lights:
                prim = stage.GetPrimAtPath(f"/World/envs/env_{env_index}/Robot/{light.name}")
                if prim.IsValid():
                    env_prims.append(prim)
            self._light_prims_by_env.append(env_prims)
        if self.device == "cpu":
            global_prims = ["/World/seabed"] if self.cfg.world.seabed_enabled else []
            self.scene.filter_collisions(global_prim_paths=global_prims)
        self.scene.rigid_objects["robot"] = self._robot
        if self._camera is not None:
            self.scene.sensors["camera"] = self._camera
        if self._camera_left is not None:
            self.scene.sensors["camera_left"] = self._camera_left
        if self._camera_right is not None:
            self.scene.sensors["camera_right"] = self._camera_right
        if self._imu is not None:
            self.scene.sensors["imu"] = self._imu
        if self._contact_sensor is not None:
            self.scene.sensors["contact"] = self._contact_sensor

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._desired_velocity_b = self._actions[:, :3] * self._max_command_velocity
        self._desired_yaw_rate = self._actions[:, 3] * self.cfg.max_command_yaw_rate_radps
        if self._light_intensity_scale.shape[1] > 0:
            expected = 4 + self._light_intensity_scale.shape[1]
            if self._actions.shape[1] < expected:
                raise ValueError(f"Expected at least {expected} action dimensions for active-light control")
            self._light_intensity_scale.copy_((self._actions[:, 4:expected] + 1.0) * 0.5)
            self._update_light_prims()

    def set_light_intensity_scale(self, intensity_scale: torch.Tensor) -> None:
        """Set independent per-environment light scales in ``[0, 1]``."""
        if self._light_intensity_scale.shape[1] == 0:
            raise RuntimeError("This task has no active-light control channels")
        scale = torch.as_tensor(intensity_scale, dtype=torch.float32, device=self.device)
        if scale.shape != self._light_intensity_scale.shape:
            raise ValueError(
                f"Expected light scale shape {tuple(self._light_intensity_scale.shape)}, "
                f"got {tuple(scale.shape)}"
            )
        self._light_intensity_scale.copy_(scale.clamp(0.0, 1.0))
        self._update_light_prims()

    def _update_light_prims(self) -> None:
        if not self._light_prims_by_env or self._light_intensity_scale.shape[1] == 0:
            return
        update_underwater_lighting_batch(
            self._light_prims_by_env,
            self.cfg.lighting,
            time_s=float(self.common_step_counter * self.step_dt),
            intensity_scales=self._light_intensity_scale,
        )

    def _apply_action(self) -> None:
        timestamp_s = float(self.common_step_counter * self.step_dt)
        control_command = ControlCommand.from_body_velocity(
            timestamp_s,
            self._desired_velocity_b.detach().clone(),
            self._desired_yaw_rate.detach().clone(),
        )
        command = command_to_thruster(
            control_command,
            velocity_controller=self._velocity_controller,
            allocator=self._allocator,
            linear_velocity_b=self._robot.data.root_lin_vel_b,
            yaw_rate=self._robot.data.root_ang_vel_b[:, 2],
        )
        if self._domain_gap_samples is not None:
            command = perturb_thruster_command(command, self._domain_gap_samples)
            assert self._domain_command_state is not None
            latency = self._domain_gap_samples["actuator_latency_s"].unsqueeze(-1)
            alpha = (self.physics_dt / (latency + self.physics_dt)).clamp(0.0, 1.0)
            self._domain_command_state += alpha * (command - self._domain_command_state)
            command = self._domain_command_state
        self._last_control_command = control_command
        self._last_thruster_command = command.detach().clone()
        self._current_field.step(self.step_dt)
        thruster_dead_zone = None
        thruster_response_time = None
        if self._domain_gap_samples is not None:
            thruster_dead_zone = self._domain_gap_samples["dead_zone"]
            thruster_response_time = self._domain_gap_samples["response_time_s"]
        thrust = self._thrusters.step(
            command,
            self.physics_dt,
            dead_zone=thruster_dead_zone,
            response_time_s=thruster_response_time,
        )
        thrust_force, thrust_torque = self._thrusters.wrench_from_thrust(thrust)
        linear_velocity_b = self._robot.data.root_lin_vel_b
        angular_velocity_b = self._robot.data.root_ang_vel_b
        linear_acceleration_b = (linear_velocity_b - self._previous_linear_velocity_b) / max(self.physics_dt, 1.0e-6)
        angular_acceleration_b = (angular_velocity_b - self._previous_angular_velocity_b) / max(self.physics_dt, 1.0e-6)
        current_velocity_w = self._current_field.velocity(timestamp_s)
        buoyancy_scale = drag_scale = center_offset = added_mass_scale = None
        if self._domain_gap_samples is not None:
            current_velocity_w = current_velocity_w.clone()
            current_velocity_w[:, 0] += self._domain_gap_samples["current_speed_mps"]
            buoyancy_scale = self._domain_gap_samples["buoyancy_scale"]
            drag_scale = self._domain_gap_samples["drag_scale"]
            center_offset = torch.zeros(self.num_envs, 3, device=self.device)
            center_offset[:, 2] = self._domain_gap_samples["center_offset_m"]
            added_mass_scale = self._domain_gap_samples["added_mass_scale"]
        hydro_force, hydro_torque = self._hydrodynamics.wrench_body(
            self._robot.data.root_quat_w,
            linear_velocity_b,
            angular_velocity_b,
            current_velocity_w=current_velocity_w,
            linear_acceleration_b=linear_acceleration_b,
            angular_acceleration_b=angular_acceleration_b,
            buoyancy_scale=buoyancy_scale,
            drag_scale=drag_scale,
            center_of_buoyancy_offset_m=center_offset,
            added_mass_scale=added_mass_scale,
        )
        self._previous_linear_velocity_b.copy_(linear_velocity_b)
        self._previous_angular_velocity_b.copy_(angular_velocity_b)
        if self._domain_gap_samples is None:
            total_force = thrust_force + hydro_force
            total_torque = thrust_torque + hydro_torque
        else:
            # PhysX mass properties are shared by Fabric-cloned bodies. Scale
            # the applied actuator wrench to emulate per-environment mass and
            # inertia variation while keeping the stable baseline body config.
            mass_scale = self._domain_gap_samples["mass_scale"].unsqueeze(-1).clamp_min(1.0e-3)
            inertia_scale = self._domain_gap_samples["inertia_scale"].unsqueeze(-1).clamp_min(1.0e-3)
            total_force = thrust_force / mass_scale + hydro_force
            total_torque = thrust_torque / inertia_scale + hydro_torque
        total_force = total_force.unsqueeze(1)
        total_torque = total_torque.unsqueeze(1)
        self._robot.instantaneous_wrench_composer.set_forces_and_torques(
            forces=total_force,
            torques=total_torque,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        ground_truth = self._ground_truth_state()
        packet = SensorPacket(namespace="batch", timestamp_s=ground_truth.timestamp_s, ground_truth=ground_truth)
        source = PolicyStateSource(self.cfg.policy_state_source)
        ground_truth_localization = GroundTruthLocalizationBackend().update(packet)
        if source is PolicyStateSource.GROUND_TRUTH:
            localization = ground_truth_localization
        elif isinstance(self._localization_backend, ExternalVIOBackend):
            localization = self._localization_backend.update_batch(self.build_sensor_packets())
        else:
            localization = self._localization_backend.update(packet)
        self._last_localization = localization
        navigation = build_navigation_observation(
            self._goal_pos_w,
            source,
            ground_truth,
            localization,
        )
        obs = navigation_observation_to_tensor(
            navigation,
            self._robot.data.projected_gravity_b,
            self._actions,
            position_scale=self.cfg.position_scale,
            linear_velocity_scale=self.cfg.linear_velocity_scale,
            angular_velocity_scale=self.cfg.angular_velocity_scale,
        )
        if not getattr(self.cfg, "visual_observation_enabled", False):
            return {"policy": obs}

        packets = self.build_sensor_packets()
        required = ("rgb_left", "rgb_right", "depth_left", "depth_right")
        if any(any(getattr(packet, name) is None for name in required) for packet in packets):
            raise RuntimeError("Visual observation requires synchronized stereo RGB/depth packets")

        stack = lambda name: torch.stack([getattr(packet, name) for packet in packets], dim=0)
        mission_command = (
            obs[:, :4]
            if bool(getattr(self.cfg, "visual_mission_command_enabled", True))
            else None
        )
        visual = build_visual_observation(
            stack("rgb_left"),
            stack("rgb_right"),
            stack("depth_left"),
            stack("depth_right"),
            imu_acceleration=None
            if any(packet.imu_acceleration is None for packet in packets)
            else stack("imu_acceleration"),
            imu_angular_velocity=None
            if any(packet.imu_angular_velocity is None for packet in packets)
            else stack("imu_angular_velocity"),
            pressure_depth_m=stack("pressure_depth_m"),
            previous_action=self._actions,
            # The goal-conditioned pilot receives the first four navigation
            # values.  The exploration task explicitly passes ``None`` so its
            # actor cannot infer a route or entrance label from a goal vector.
            mission_command=mission_command,
            output_hw=tuple(getattr(self.cfg, "visual_output_hw", (12, 16))),
            max_depth_m=float(getattr(self.cfg, "visual_max_depth_m", 12.0)),
        )
        expected_dim = int(self.cfg.observation_space)
        if visual.shape[-1] != expected_dim:
            raise RuntimeError(f"Visual feature dimension mismatch: {visual.shape[-1]} != {expected_dim}")
        critic = (
            self._exploration_critic_observation(obs, ground_truth)
            if bool(getattr(self.cfg, "exploration_reward_enabled", False))
            else obs
        )
        return {"policy": visual, "critic": critic}

    def _exploration_critic_observation(
        self,
        navigation_tensor: torch.Tensor,
        ground_truth: GroundTruthState,
    ) -> torch.Tensor:
        """Build privileged state without a goal or centerline coordinate."""
        workspace = getattr(self.cfg.world, "navigation_workspace_size_m", None) or self.cfg.workspace_size_m
        half_workspace = 0.5 * torch.tensor(workspace, dtype=torch.float32, device=self.device)
        relative_position = ground_truth.position_w - self.scene.env_origins
        normalized_position = (relative_position / half_workspace.clamp_min(1.0e-6)).clamp(-2.0, 2.0)
        if self._exploration_tracker is None:
            workspace_coverage = torch.zeros((self.num_envs, 1), device=self.device)
        else:
            workspace_coverage = self._exploration_tracker.workspace_coverage_fraction.unsqueeze(-1)
        normalized_clearance = (
            self._exploration_forward_clearance_m / float(self.cfg.visual_max_depth_m)
        ).clamp(0.0, 1.0).unsqueeze(-1)
        normalized_contact = (
            self._cave_contact_force_n / max(float(self.cfg.collision_force_threshold_n), 1.0e-6)
        ).clamp(0.0, 10.0).unsqueeze(-1)
        # navigation_tensor[4:] contains body velocities, projected gravity,
        # and the previous action.  Its goal direction/distance are omitted.
        features = [
            normalized_position,
            navigation_tensor[:, 4:],
            workspace_coverage,
            normalized_clearance,
            normalized_contact,
        ]
        if bool(getattr(self.cfg, "cave_entry_enabled", False)):
            distance_scale = max(
                float(self.cfg.cave_entry_spawn_distance_range_m[1]),
                float(self.cfg.cave_entry_depth_m),
                1.0e-6,
            )
            gate_scale = max(float(self.cfg.cave_entry_gate_radius_m), 1.0e-6)
            features.append(
                torch.stack(
                    (
                        (self._cave_entry_signed_depth_m / distance_scale).clamp(-2.0, 2.0),
                        (self._cave_entry_radial_distance_m / gate_scale).clamp(0.0, 2.0),
                        self._episode_entered_cave.float(),
                    ),
                    dim=-1,
                )
            )
        return torch.cat(features, dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        self._refresh_cave_metrics()
        goal_vector_b = rotate_world_to_body(
            self._robot.data.root_quat_w,
            self._goal_pos_w - self._robot.data.root_pos_w,
        )
        distance = torch.linalg.vector_norm(goal_vector_b, dim=-1)
        progress = self._previous_distance - distance
        self._previous_distance = distance
        heading = goal_vector_b[:, 0] / distance.clamp_min(1.0e-6)
        rewards = {
            "progress": self.cfg.progress_weight * progress,
            "goal": self.cfg.goal_bonus * self._success.float(),
            "heading": self.cfg.heading_weight * heading,
            "action": -self.cfg.action_penalty_weight * self._actions.square().sum(dim=-1),
            "bounds": -self.cfg.out_of_bounds_penalty * self._out_of_bounds.float(),
            "route_progress": torch.zeros_like(progress),
            "route_deviation": torch.zeros_like(progress),
            "clearance": torch.zeros_like(progress),
            "collision": torch.zeros_like(progress),
            "exploration": torch.zeros_like(progress),
            "exploration_clearance": torch.zeros_like(progress),
            "cave_entry": torch.zeros_like(progress),
        }
        if self._has_cave_route and getattr(self.cfg, "cave_route_reward_enabled", False):
            route_progress = self._cave_route_chainage - self._previous_cave_route_chainage
            self._previous_cave_route_chainage = self._cave_route_chainage.detach().clone()
            route_deviation = torch.relu(self._cave_centerline_distance - float(self.cfg.route_deviation_soft_m))
            clearance_required = float(self.cfg.vehicle_bounding_radius_m) + float(self.cfg.clearance_margin_m)
            # A missing/NaN clearance annotation is deliberately neutral.  It
            # must not silently become a false safety signal.
            clearance_risk = torch.where(
                torch.isfinite(self._cave_local_clearance),
                torch.relu(clearance_required - self._cave_local_clearance),
                torch.zeros_like(progress),
            )
            rewards["route_progress"] = float(self.cfg.route_progress_weight) * route_progress
            rewards["route_deviation"] = -float(self.cfg.route_deviation_penalty_weight) * route_deviation.square()
            rewards["clearance"] = -float(self.cfg.centerline_clearance_penalty_weight) * clearance_risk.square()
            rewards["collision"] = -float(self.cfg.collision_penalty) * self._cave_collision.float()
        if self._exploration_tracker is not None:
            rewards["exploration"] = (
                float(self.cfg.exploration_new_voxel_reward) * self._exploration_tracker.new_voxel.float()
                - float(self.cfg.exploration_revisit_penalty)
                * (~self._exploration_tracker.new_voxel & self._exploration_tracker.position_valid).float()
            )
            clearance_risk = torch.relu(
                float(self.cfg.exploration_clearance_margin_m) - self._exploration_forward_clearance_m
            )
            rewards["exploration_clearance"] = (
                -float(self.cfg.exploration_clearance_penalty_weight) * clearance_risk.square()
            )
            rewards["collision"] = -float(self.cfg.collision_penalty) * self._cave_collision.float()
        if bool(getattr(self.cfg, "cave_entry_enabled", False)):
            rewards["cave_entry"] = float(self.cfg.cave_entry_bonus) * self._cave_entry_event.float()
        for key, value in rewards.items():
            self._episode_sums[key] += value
        total_reward = torch.stack(tuple(rewards.values())).sum(dim=0)
        self._write_episode_records(total_reward)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_episode_path_length()
        self._refresh_cave_metrics()
        self._refresh_exploration_metrics()
        self._refresh_cave_entry_metrics()
        relative_pos = self._robot.data.root_pos_w - self.scene.env_origins
        workspace = getattr(self.cfg.world, "navigation_workspace_size_m", None) or self.cfg.workspace_size_m
        half_x, half_y, half_z = (float(value) * 0.5 for value in workspace)
        self._out_of_bounds = (relative_pos[:, 0].abs() > half_x) | (relative_pos[:, 1].abs() > half_y)
        if not self._has_cave_route:
            self._out_of_bounds |= (relative_pos[:, 2] < 0.25) | (relative_pos[:, 2] > float(workspace[2]))
        else:
            self._out_of_bounds |= relative_pos[:, 2].abs() > half_z
        collision_termination = torch.zeros_like(self._cave_collision)
        if self._has_cave_route and getattr(self.cfg, "cave_route_reward_enabled", False):
            self._out_of_bounds |= self._cave_centerline_distance > float(self.cfg.route_deviation_hard_m)
            if getattr(self.cfg, "cave_contact_termination_enabled", False):
                collision_termination |= self._cave_collision
        if bool(getattr(self.cfg, "cave_entry_enabled", False)):
            if getattr(self.cfg, "cave_contact_termination_enabled", False):
                collision_termination |= self._cave_collision
            self._success.copy_(self._episode_entered_cave)
        elif self._exploration_tracker is None:
            distance = torch.linalg.vector_norm(self._goal_pos_w - self._robot.data.root_pos_w, dim=-1)
            self._success = distance < self.cfg.goal_threshold_m
        else:
            # Coverage has no single privileged goal pose.  Evaluation uses
            # unique voxels and workspace coverage rather than goal success.
            self._success.zero_()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._out_of_bounds | collision_termination
        if not bool(getattr(self.cfg, "cave_entry_enabled", False)) or bool(
            getattr(self.cfg, "cave_entry_terminate_on_success", True)
        ):
            terminated |= self._success
        return terminated, time_out

    def _refresh_cave_metrics(self) -> None:
        """Update route/contact metrics with batched tensors.

        The centerline is an external provisional route annotation.  It is
        used for teacher shaping and diagnostics, never exposed to the visual
        actor as an observation or treated as proof of free-space connectivity.
        """
        if not self._has_cave_route:
            self._cave_collision.zero_()
            self._cave_contact_force_n.zero_()
            self._cave_centerline_distance.zero_()
            self._cave_route_chainage.zero_()
            self._cave_local_clearance.fill_(float("inf"))
            return

        relative_pos = self._robot.data.root_pos_w - self.scene.env_origins
        if self._multi_cave_centerlines is not None:
            assert self._multi_cave_chainages is not None
            assert self._multi_cave_route_mask is not None
            assert self._multi_cave_clearances is not None
            centerlines = self._multi_cave_centerlines[self._cave_scene_ids]
            route_mask = self._multi_cave_route_mask[self._cave_scene_ids]
            distances_sq = torch.sum(
                (relative_pos.unsqueeze(1) - centerlines).square(), dim=-1
            ).masked_fill(~route_mask, float("inf"))
            nearest_distance_sq, nearest_index = distances_sq.min(dim=1)
            selected_chainages = self._multi_cave_chainages[self._cave_scene_ids]
            selected_clearances = self._multi_cave_clearances[self._cave_scene_ids]
            self._cave_route_chainage = selected_chainages.gather(
                1, nearest_index.unsqueeze(-1)
            ).squeeze(-1)
            self._cave_local_clearance = selected_clearances.gather(
                1, nearest_index.unsqueeze(-1)
            ).squeeze(-1)
        else:
            assert self._cave_centerline is not None and self._cave_chainage is not None
            distances_sq = torch.sum(
                (relative_pos.unsqueeze(1) - self._cave_centerline.unsqueeze(0)).square(), dim=-1
            )
            nearest_distance_sq, nearest_index = distances_sq.min(dim=1)
            self._cave_route_chainage = self._cave_chainage[nearest_index]
            if self._cave_surface_clearance is not None:
                self._cave_local_clearance = self._cave_surface_clearance[nearest_index]
            else:
                self._cave_local_clearance.fill_(float("inf"))
        self._cave_centerline_distance = torch.sqrt(nearest_distance_sq.clamp_min(0.0))
        self._cave_contact_force_n.zero_()
        if self._contact_sensor is not None:
            net_forces = self._contact_sensor.data.net_forces_w
            if net_forces is not None:
                self._cave_contact_force_n = torch.linalg.vector_norm(net_forces, dim=-1).amax(dim=-1)
        self._cave_collision = self._cave_contact_force_n > float(self.cfg.collision_force_threshold_n)
        self._episode_had_collision |= self._cave_collision

    def _refresh_episode_path_length(self) -> None:
        position = self._robot.data.root_pos_w
        displacement = torch.linalg.vector_norm(position - self._previous_episode_position_w, dim=-1)
        self._episode_path_length_m += displacement
        self._previous_episode_position_w.copy_(position)

    def _refresh_exploration_metrics(self) -> None:
        if self._exploration_tracker is None:
            return
        relative_position = self._robot.data.root_pos_w - self.scene.env_origins
        self._exploration_tracker.update(relative_position)

        clearances = []
        for camera in (self._camera_left, self._camera_right):
            if camera is None:
                continue
            depth = camera.data.output.get("depth")
            if depth is None:
                continue
            clearances.append(
                robust_forward_clearance(
                    depth,
                    max_depth_m=float(self.cfg.visual_max_depth_m),
                    crop_fraction=float(self.cfg.exploration_depth_crop_fraction),
                    quantile=float(self.cfg.exploration_depth_quantile),
                )
            )
        if clearances:
            self._exploration_forward_clearance_m = torch.stack(clearances, dim=0).amin(dim=0)
        else:
            self._exploration_forward_clearance_m.fill_(float(self.cfg.visual_max_depth_m))

    def _refresh_cave_entry_metrics(self) -> None:
        """Update the automatically inferred portal-crossing event."""
        if not bool(getattr(self.cfg, "cave_entry_enabled", False)):
            self._cave_entry_event.zero_()
            return
        relative_position = self._robot.data.root_pos_w - self.scene.env_origins
        portal_position = self._cave_portal_positions[self._cave_entry_portal_id]
        inward = self._cave_portal_directions[self._cave_entry_portal_id]
        delta = relative_position - portal_position
        signed_depth = torch.sum(delta * inward, dim=-1)
        radial = torch.linalg.vector_norm(delta - signed_depth.unsqueeze(-1) * inward, dim=-1)
        inside_gate = (signed_depth >= float(self.cfg.cave_entry_depth_m)) & (
            radial <= float(self.cfg.cave_entry_gate_radius_m)
        )
        self._cave_entry_signed_depth_m.copy_(signed_depth)
        self._cave_entry_radial_distance_m.copy_(radial)
        self._cave_entry_event.copy_(inside_gate & ~self._episode_entered_cave)
        self._episode_entered_cave |= inside_gate

    def _sample_cave_entry_spawn(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an exterior pose from inferred portal geometry."""
        assert self._cave_centerline is not None and self._cave_chainage is not None
        count = env_ids.numel()
        portal_ids = torch.randint(len(self._cave_portals), (count,), device=self.device)
        self._cave_entry_portal_id[env_ids] = portal_ids
        minimum, maximum = (float(value) for value in self.cfg.cave_entry_spawn_distance_range_m)
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError("cave_entry_spawn_distance_range_m must be positive and ordered")
        available = torch.tensor(
            [portal.exterior_run_m for portal in self._cave_portals],
            dtype=torch.float32,
            device=self.device,
        )[portal_ids]
        if bool(torch.any(available < minimum)):
            raise ValueError("Inferred portal exterior run is shorter than the minimum spawn distance")
        upper = torch.minimum(torch.full_like(available, maximum), available)
        distance = minimum + torch.rand(count, device=self.device) * (upper - minimum)
        query_chainage = self._cave_portal_chainages[portal_ids] - (
            self._cave_portal_chainage_directions[portal_ids] * distance
        )
        spawn = interpolate_polyline(self._cave_centerline, self._cave_chainage, query_chainage)

        inward = self._cave_portal_directions[portal_ids]
        lateral = torch.stack((-inward[:, 1], inward[:, 0], torch.zeros_like(inward[:, 0])), dim=-1)
        lateral = lateral / torch.linalg.vector_norm(lateral, dim=-1, keepdim=True).clamp_min(1.0e-6)
        lateral_jitter = float(self.cfg.cave_entry_spawn_lateral_jitter_m)
        vertical_jitter = float(self.cfg.cave_entry_spawn_vertical_jitter_m)
        if lateral_jitter > 0.0:
            spawn += lateral * torch.empty(count, 1, device=self.device).uniform_(
                -lateral_jitter, lateral_jitter
            )
        if vertical_jitter > 0.0:
            spawn[:, 2] += torch.empty(count, device=self.device).uniform_(
                -vertical_jitter, vertical_jitter
            )
        yaw = torch.atan2(inward[:, 1], inward[:, 0])
        if bool(getattr(self.cfg, "exploration_random_yaw", False)):
            yaw.uniform_(-math.pi, math.pi)
        return spawn, yaw, query_chainage

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if env_ids.numel() > 0:
            if self._exploration_tracker is not None:
                self._last_episode_workspace_coverage[env_ids] = (
                    self._exploration_tracker.workspace_coverage_fraction[env_ids]
                )
                self._last_episode_unique_voxels[env_ids] = self._exploration_tracker.visited_count[env_ids]
                self._last_episode_path_length_m[env_ids] = self._exploration_tracker.path_length_m[env_ids]
            else:
                self._last_episode_path_length_m[env_ids] = self._episode_path_length_m[env_ids]
            if self._multi_cave_centerlines is not None:
                shortest_path = self._cave_scene_route_lengths[self._cave_scene_ids[env_ids]]
            elif self._cave_chainage is not None:
                spawn_chainage = float(getattr(self.cfg.world, "spawn_chainage_m", 0.0))
                goal_chainage = float(getattr(self.cfg.world, "goal_chainage_m", spawn_chainage))
                shortest_path = torch.full(
                    (env_ids.numel(),),
                    abs(goal_chainage - spawn_chainage),
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                shortest_path = torch.zeros(env_ids.numel(), device=self.device)
            self._last_episode_reference_path_m[env_ids] = shortest_path
            valid_path = shortest_path > 0.0
            self._last_episode_spl[env_ids] = torch.where(
                self._success[env_ids] & valid_path,
                shortest_path
                / torch.maximum(shortest_path, self._last_episode_path_length_m[env_ids]),
                torch.zeros_like(shortest_path),
            )
            self._last_episode_success[env_ids] = self._success[env_ids]
            self._last_episode_out_of_bounds[env_ids] = self._out_of_bounds[env_ids]
            self._last_episode_timeout[env_ids] = self.reset_time_outs[env_ids]
            self._last_episode_collision[env_ids] = self._episode_had_collision[env_ids]
            self.extras["log"] = {
                f"Episode_Reward/{key}": self._episode_sums[key][env_ids].mean().item()
                for key in self._episode_sums
            }
            self.extras["log"]["Metrics/success_rate"] = self._success[env_ids].float().mean().item()
            self.extras["log"]["Metrics/collision_rate"] = self._episode_had_collision[env_ids].float().mean().item()
            self.extras["log"]["Metrics/route_deviation_m"] = self._cave_centerline_distance[env_ids].mean().item()
            self.extras["log"]["Metrics/contact_force_n"] = self._cave_contact_force_n[env_ids].mean().item()
            self.extras["episode_success"] = self._last_episode_success.clone()
            self.extras["episode_out_of_bounds"] = self._last_episode_out_of_bounds.clone()
            self.extras["episode_timeout"] = self._last_episode_timeout.clone()
            self.extras["episode_collision"] = self._last_episode_collision.clone()
            self.extras["episode_workspace_coverage"] = self._last_episode_workspace_coverage.clone()
            self.extras["episode_unique_voxels"] = self._last_episode_unique_voxels.clone()
            self.extras["episode_path_length_m"] = self._last_episode_path_length_m.clone()
            self.extras["episode_reference_path_m"] = self._last_episode_reference_path_m.clone()
            self.extras["episode_spl"] = self._last_episode_spl.clone()
            self.extras["episode_scene_id"] = self._cave_scene_ids.clone()
            if self._has_cave_route:
                self.extras["log"]["Metrics/SPL"] = self._last_episode_spl[env_ids].mean().item()
                self.extras["log"]["Metrics/path_length_m"] = (
                    self._last_episode_path_length_m[env_ids].mean().item()
                )
            if self._cave_scene_variants:
                completed_scene_ids = self._cave_scene_ids[env_ids]
                for scene_id, scene in enumerate(self._cave_scene_variants):
                    scene_mask = completed_scene_ids == scene_id
                    if bool(torch.any(scene_mask)):
                        selected_ids = env_ids[scene_mask]
                        prefix = f"Scene/{scene.key}"
                        self.extras["log"][f"{prefix}/success_rate"] = (
                            self._success[selected_ids].float().mean().item()
                        )
                        self.extras["log"][f"{prefix}/collision_rate"] = (
                            self._episode_had_collision[selected_ids].float().mean().item()
                        )
                        self.extras["log"][f"{prefix}/SPL"] = (
                            self._last_episode_spl[selected_ids].mean().item()
                        )
            if self._exploration_tracker is not None:
                self.extras["log"]["Metrics/workspace_coverage_fraction"] = (
                    self._last_episode_workspace_coverage[env_ids].mean().item()
                )
                self.extras["log"]["Metrics/unique_voxels"] = (
                    self._last_episode_unique_voxels[env_ids].float().mean().item()
                )
                self.extras["log"]["Metrics/path_length_m"] = (
                    self._last_episode_path_length_m[env_ids].mean().item()
                )
            if bool(getattr(self.cfg, "cave_entry_enabled", False)):
                self.extras["log"]["Metrics/cave_entry_rate"] = (
                    self._episode_entered_cave[env_ids].float().mean().item()
                )
                self.extras["log"]["Metrics/entry_signed_depth_m"] = (
                    self._cave_entry_signed_depth_m[env_ids].mean().item()
                )
            for values in self._episode_sums.values():
                values[env_ids] = 0.0

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._thrusters.reset(self.num_envs, env_ids)
        self._current_field.reset(env_ids)
        self._previous_linear_velocity_b[env_ids] = 0.0
        self._previous_angular_velocity_b[env_ids] = 0.0
        if self._domain_command_state is not None:
            self._domain_command_state[env_ids] = 0.0
        if self._domain_gap_cfg is not None and self._domain_gap_samples is not None:
            sampled = sample_domain_gap(self._domain_gap_cfg, env_ids.numel(), device=self.device)
            for key, value in sampled.items():
                self._domain_gap_samples[key][env_ids] = value
        self._actions[env_ids] = 0.0
        self._desired_velocity_b[env_ids] = 0.0
        self._desired_yaw_rate[env_ids] = 0.0
        if self._light_intensity_scale.shape[1] > 0:
            self._light_intensity_scale[env_ids] = float(getattr(self.cfg, "light_control_default_scale", 1.0))

        count = env_ids.numel()
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        spawn_route_chainage: torch.Tensor | None = None
        if self._multi_cave_centerlines is not None:
            scene_ids = self._cave_scene_ids[env_ids]
            spawn = self._cave_scene_spawn_points[scene_ids].clone()
            jitter_scale = self._cave_scene_spawn_jitters[scene_ids].unsqueeze(-1)
            if bool(torch.any(jitter_scale > 0.0)):
                spawn += torch.randn_like(spawn) * jitter_scale
            root_state[:, :3] = spawn + self.scene.env_origins[env_ids]
            self._goal_pos_w[env_ids] = self._cave_scene_goal_points[scene_ids] + self.scene.env_origins[env_ids]
            spawn_route_chainage = self._cave_scene_spawn_chainages[scene_ids].clone()
            tangent = self._cave_scene_spawn_tangents[scene_ids]
            yaw = torch.atan2(tangent[:, 1], tangent[:, 0])
            yaw_jitter = float(getattr(self.cfg, "cave_spawn_yaw_jitter_rad", 0.0))
            if yaw_jitter > 0.0:
                yaw += torch.empty(count, device=self.device).uniform_(-yaw_jitter, yaw_jitter)
        elif self._cave_centerline is None:
            root_state[:, :2] += torch.empty(count, 2, device=self.device).uniform_(
                -self.cfg.reset_xy_m, self.cfg.reset_xy_m
            )
            root_state[:, 2] = torch.empty(count, device=self.device).uniform_(*self.cfg.reset_depth_range_m)
            yaw = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        elif bool(getattr(self.cfg, "cave_entry_enabled", False)):
            spawn, yaw, spawn_route_chainage = self._sample_cave_entry_spawn(env_ids)
            root_state[:, :3] = spawn + self.scene.env_origins[env_ids]
            # The entry task has no goal.  Keep the legacy storage benign so
            # accidental future use cannot encode an inferred portal vector.
            self._goal_pos_w[env_ids] = root_state[:, :3]
        else:
            assert self._cave_chainage is not None
            spawn_chainage = float(getattr(self.cfg.world, "spawn_chainage_m", 2.0))
            goal_chainage = float(getattr(self.cfg.world, "goal_chainage_m", spawn_chainage + 10.0))
            spawn_idx = torch.argmin((self._cave_chainage - spawn_chainage).abs())
            goal_idx = torch.argmin((self._cave_chainage - goal_chainage).abs())
            spawn = self._cave_centerline[spawn_idx].expand(count, -1).clone()
            spawn_route_chainage = torch.full(
                (count,), float(self._cave_chainage[spawn_idx]), device=self.device
            )
            if float(getattr(self.cfg.world, "spawn_jitter_m", 0.0)) > 0.0:
                spawn += torch.randn_like(spawn) * float(self.cfg.world.spawn_jitter_m)
            root_state[:, :3] = spawn + self.scene.env_origins[env_ids]
            goal = self._cave_centerline[goal_idx].expand(count, -1)
            self._goal_pos_w[env_ids] = goal + self.scene.env_origins[env_ids]
            next_idx = min(int(spawn_idx.item()) + 1, self._cave_centerline.shape[0] - 1)
            tangent = self._cave_centerline[next_idx] - self._cave_centerline[spawn_idx]
            yaw = torch.full((count,), torch.atan2(tangent[1], tangent[0]).item(), device=self.device)
            if bool(getattr(self.cfg, "exploration_random_yaw", False)):
                yaw.uniform_(-math.pi, math.pi)
            else:
                yaw_jitter = float(getattr(self.cfg, "cave_spawn_yaw_jitter_rad", 0.0))
                if yaw_jitter > 0.0:
                    yaw += torch.empty(count, device=self.device).uniform_(-yaw_jitter, yaw_jitter)
        root_state[:, 3:7] = 0.0
        root_state[:, 3] = torch.cos(0.5 * yaw)
        root_state[:, 6] = torch.sin(0.5 * yaw)
        root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        if not self._has_cave_route:
            angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
            radius = torch.empty(count, device=self.device).uniform_(*self.cfg.goal_distance_range_m)
            self._goal_pos_w[env_ids, 0] = self.scene.env_origins[env_ids, 0] + radius * torch.cos(angle)
            self._goal_pos_w[env_ids, 1] = self.scene.env_origins[env_ids, 1] + radius * torch.sin(angle)
            vertical_offset = torch.empty(count, device=self.device).uniform_(
                -self.cfg.goal_vertical_offset_m, self.cfg.goal_vertical_offset_m
            )
            self._goal_pos_w[env_ids, 2] = (root_state[:, 2] + vertical_offset).clamp(
                0.5, float(self.cfg.workspace_size_m[2]) - 0.5
            )
        self._previous_distance[env_ids] = torch.linalg.vector_norm(
            self._goal_pos_w[env_ids] - root_state[:, :3], dim=-1
        )
        self._episode_path_length_m[env_ids] = 0.0
        self._previous_episode_position_w[env_ids] = root_state[:, :3]
        self._success[env_ids] = False
        self._out_of_bounds[env_ids] = False
        self._cave_collision[env_ids] = False
        self._episode_had_collision[env_ids] = False
        self._last_episode_collision[env_ids] = False
        self._cave_contact_force_n[env_ids] = 0.0
        self._cave_centerline_distance[env_ids] = 0.0
        self._cave_route_chainage[env_ids] = 0.0
        self._previous_cave_route_chainage[env_ids] = 0.0
        self._cave_local_clearance[env_ids] = float("inf")
        self._exploration_forward_clearance_m[env_ids] = float(self.cfg.visual_max_depth_m)
        self._cave_entry_event[env_ids] = False
        self._episode_entered_cave[env_ids] = False
        if spawn_route_chainage is not None:
            self._previous_cave_route_chainage[env_ids] = spawn_route_chainage
            self._cave_route_chainage[env_ids] = spawn_route_chainage
        if bool(getattr(self.cfg, "cave_entry_enabled", False)):
            portal_position = self._cave_portal_positions[self._cave_entry_portal_id[env_ids]]
            inward = self._cave_portal_directions[self._cave_entry_portal_id[env_ids]]
            delta = root_state[:, :3] - self.scene.env_origins[env_ids] - portal_position
            signed_depth = torch.sum(delta * inward, dim=-1)
            self._cave_entry_signed_depth_m[env_ids] = signed_depth
            self._cave_entry_radial_distance_m[env_ids] = torch.linalg.vector_norm(
                delta - signed_depth.unsqueeze(-1) * inward, dim=-1
            )
        else:
            self._cave_entry_signed_depth_m[env_ids] = 0.0
            self._cave_entry_radial_distance_m[env_ids] = 0.0
        if self._exploration_tracker is not None:
            self._exploration_tracker.reset(
                env_ids,
                root_state[:, :3] - self.scene.env_origins[env_ids],
            )
        self._update_light_prims()

    def _ground_truth_state(self) -> GroundTruthState:
        timestamp_s = float(self.common_step_counter * self.step_dt)
        return GroundTruthState(
            timestamp_s=timestamp_s,
            position_w=self._robot.data.root_pos_w,
            orientation_wxyz=self._robot.data.root_quat_w,
            linear_velocity_w=self._robot.data.root_lin_vel_w,
            angular_velocity_w=self._robot.data.root_ang_vel_w,
        )

    def build_sensor_packets(self, rewards: torch.Tensor | None = None) -> list[SensorPacket]:
        """Return synchronized per-robot packets for perception/VIO adapters."""
        ground_truth = self._ground_truth_state()
        timestamp_s = ground_truth.timestamp_s
        rgb = depth = imu_acceleration = imu_angular_velocity = None
        rgb_left = rgb_right = depth_left = depth_right = None
        if self._camera is not None:
            rgb = self._camera.data.output.get("rgb")
            depth = self._camera.data.output.get("depth")
        if self._camera_left is not None:
            rgb_left = self._camera_left.data.output.get("rgb")
            depth_left = self._camera_left.data.output.get("depth")
            # Preserve the established mono packet contract: ``rgb/depth``
            # refer to the left camera whenever stereo mode is active.
            rgb = rgb_left
            depth = depth_left
        if self._camera_right is not None:
            rgb_right = self._camera_right.data.output.get("rgb")
            depth_right = self._camera_right.data.output.get("depth")
        appearance_kwargs = {}
        if self._domain_gap_samples is not None:
            appearance_kwargs = {
                "visibility_range_m": self._domain_gap_samples["visibility_range_m"],
                "attenuation_scale": self._domain_gap_samples["attenuation_scale"],
                "backscatter_strength": self._domain_gap_samples["backscatter_strength"],
                "exposure_offset": self._domain_gap_samples["exposure_offset"],
                "motion_blur_strength": self._domain_gap_samples["motion_blur_strength"],
            }
        if rgb is not None and depth is not None:
            rgb = apply_underwater_appearance(rgb, depth, self.cfg.appearance, **appearance_kwargs)
        if rgb_right is not None and depth_right is not None:
            rgb_right = apply_underwater_appearance(rgb_right, depth_right, self.cfg.appearance, **appearance_kwargs)
        if self._domain_gap_samples is not None:
            rgb_std = self._domain_gap_samples["camera_noise_std"]
            depth_std = self._domain_gap_samples["depth_noise_std"]
            for image_name in ("rgb", "rgb_right"):
                image = locals()[image_name]
                if image is None:
                    continue
                noise = torch.randn(image.shape, device=self.device)
                per_env_std = rgb_std.reshape(self.num_envs, *([1] * (image.ndim - 1)))
                if image.dtype == torch.uint8:
                    image = (image.float() + noise * per_env_std * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                else:
                    image = (image.float() + noise * per_env_std).clamp(0.0, 1.0).to(image.dtype)
                if image_name == "rgb":
                    rgb = image
                else:
                    rgb_right = image
            for depth_name in ("depth", "depth_right"):
                image = locals()[depth_name]
                if image is None:
                    continue
                noise = torch.randn(image.shape, device=self.device)
                per_env_std = depth_std.reshape(self.num_envs, *([1] * (image.ndim - 1)))
                image = (image.float() + noise * per_env_std).clamp_min(0.0).to(image.dtype)
                if depth_name == "depth":
                    depth = image
                else:
                    depth_right = image
        if self._camera_left is not None:
            rgb_left, depth_left = rgb, depth
        if self._imu is not None:
            imu_acceleration = self._imu.data.lin_acc_b
            imu_angular_velocity = self._imu.data.ang_vel_b
            if self._domain_gap_samples is not None:
                imu_shape = self._domain_gap_samples["imu_noise_std"].reshape(self.num_envs, 1)
                imu_bias = self._domain_gap_samples["imu_bias_std"].reshape(self.num_envs, 1)
                imu_acceleration = imu_acceleration + torch.randn_like(imu_acceleration) * imu_shape
                imu_acceleration = imu_acceleration + torch.randn_like(imu_acceleration) * imu_bias
                imu_angular_velocity = imu_angular_velocity + torch.randn_like(imu_angular_velocity) * imu_shape
                imu_acceleration = rotate_body_vectors_z(
                    imu_acceleration, self._domain_gap_samples["extrinsic_rotation_deg"]
                )
                imu_angular_velocity = rotate_body_vectors_z(
                    imu_angular_velocity, self._domain_gap_samples["extrinsic_rotation_deg"]
                )
        pressure_depth = (ground_truth.position_w[:, 2] - self.scene.env_origins[:, 2]).unsqueeze(-1)
        if self._domain_gap_samples is not None:
            pressure_depth = pressure_depth + torch.randn_like(pressure_depth) * self._domain_gap_samples[
                "depth_noise_std"
            ].unsqueeze(-1)
        jitter = torch.zeros(self.num_envs, device=self.device)
        frame_dropped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self._domain_gap_samples is not None:
            jitter = torch.randn(self.num_envs, device=self.device) * self._domain_gap_samples["timestamp_jitter_s"]
            frame_dropped = torch.rand(self.num_envs, device=self.device) < self._domain_gap_samples[
                "frame_drop_probability"
            ]
            # A recurrent control policy needs a fixed tensor every step.
            # Model a dropped frame as the common camera-driver behaviour of
            # holding the last image, while metadata still reports the drop.
            for name, image in (
                ("rgb", rgb),
                ("depth", depth),
                ("rgb_right", rgb_right),
                ("depth_right", depth_right),
            ):
                if image is None:
                    continue
                previous = self._held_sensor_frames.get(name)
                if previous is None or previous.shape != image.shape:
                    previous = image.detach().clone()
                    self._held_sensor_frames[name] = previous
                mask = frame_dropped.reshape(self.num_envs, *([1] * (image.ndim - 1)))
                held = torch.where(mask, previous, image)
                previous.copy_(held)
                if name == "rgb":
                    rgb = held
                elif name == "depth":
                    depth = held
                elif name == "rgb_right":
                    rgb_right = held
                else:
                    depth_right = held
            if self._camera_left is not None:
                rgb_left, depth_left = rgb, depth
        packets = []
        for robot_index in range(self.num_envs):
            packet_timestamp = timestamp_s + float(jitter[robot_index].item())
            camera_imu_offset_s = 0.0
            extrinsic_rotation_deg = 0.0
            exposure_offset = 0.0
            motion_blur_strength = 0.0
            if self._domain_gap_samples is not None:
                camera_imu_offset_s = float(
                    self._domain_gap_samples["camera_imu_offset_s"][robot_index].item()
                )
                extrinsic_rotation_deg = float(
                    self._domain_gap_samples["extrinsic_rotation_deg"][robot_index].item()
                )
                exposure_offset = float(self._domain_gap_samples["exposure_offset"][robot_index].item())
                motion_blur_strength = float(
                    self._domain_gap_samples["motion_blur_strength"][robot_index].item()
                )
            dropped = bool(frame_dropped[robot_index].item())
            robot_command = None
            if self._last_control_command is not None:
                command = self._last_control_command
                robot_command = ControlCommand.from_body_velocity(
                    command.timestamp_s,
                    command.body_velocity[robot_index],
                    command.yaw_rate[robot_index],
                )
            packets.append(
                SensorPacket(
                    namespace=f"robot_{robot_index:03d}",
                    timestamp_s=packet_timestamp,
                    rgb=None if rgb is None else rgb[robot_index],
                    depth=None if depth is None else depth[robot_index],
                    rgb_left=None if rgb_left is None else rgb_left[robot_index],
                    rgb_right=None if rgb_right is None else rgb_right[robot_index],
                    depth_left=None if depth_left is None else depth_left[robot_index],
                    depth_right=None if depth_right is None else depth_right[robot_index],
                    imu_acceleration=None if imu_acceleration is None else imu_acceleration[robot_index],
                    imu_angular_velocity=None if imu_angular_velocity is None else imu_angular_velocity[robot_index],
                    pressure_depth_m=pressure_depth[robot_index],
                    ground_truth=GroundTruthState(
                        timestamp_s=timestamp_s,
                        position_w=ground_truth.position_w[robot_index],
                        orientation_wxyz=ground_truth.orientation_wxyz[robot_index],
                        linear_velocity_w=ground_truth.linear_velocity_w[robot_index],
                        angular_velocity_w=ground_truth.angular_velocity_w[robot_index],
                    ),
                    control_command=robot_command,
                    thruster_command=(
                        None
                        if self._last_thruster_command is None
                        else self._last_thruster_command[robot_index]
                    ),
                    sensor_metadata={
                        "rgb": rgb is not None and not dropped,
                        "depth": depth is not None and not dropped,
                        "stereo": rgb_left is not None and rgb_right is not None,
                        "rgb_left": rgb_left is not None and not dropped,
                        "rgb_right": rgb_right is not None and not dropped,
                        "depth_left": depth_left is not None and not dropped,
                        "depth_right": depth_right is not None and not dropped,
                        "stereo_baseline_m": getattr(self.cfg, "stereo_baseline_m", None),
                        "stereo_baseline_source": "bluerov2.scn position-derived candidate; baseline attribute unresolved",
                        "stereo_near_clip_m": getattr(self.cfg, "stereo_near_clip_m", None),
                        "stereo_near_clip_reason": "estimated self-occlusion guard for opaque visual hull",
                        "imu": imu_acceleration is not None,
                        "pressure": True,
                        "ground_truth": True,
                        "contact_force_n": float(self._cave_contact_force_n[robot_index].item()),
                        "contact_proxy": "unfiltered_static_collider" if self._contact_sensor is not None else "disabled",
                        "timestamp_tolerance_s": 1.0e-4,
                        "timestamp_jitter_s": float(jitter[robot_index].item()),
                        "frame_dropped": dropped,
                        "imu_timestamp_s": packet_timestamp,
                        "camera_timestamp_s": packet_timestamp + camera_imu_offset_s,
                        "camera_imu_offset_s": camera_imu_offset_s,
                        "extrinsic_rotation_deg": extrinsic_rotation_deg,
                        "exposure_offset": exposure_offset,
                        "motion_blur_strength": motion_blur_strength,
                        "domain_randomization": self._domain_gap_metadata(robot_index),
                    },
                    collision=bool((self._out_of_bounds[robot_index] | self._cave_collision[robot_index]).item()),
                    goal_position_w=self._goal_pos_w[robot_index],
                    reward=None if rewards is None else rewards[robot_index],
                )
            )
        return packets

    def _domain_gap_metadata(self, robot_index: int) -> dict[str, float]:
        if self._domain_gap_samples is None:
            return {}
        return {
            key: float(value[robot_index].item())
            for key, value in self._domain_gap_samples.items()
            if value.ndim == 1
        }

    def _write_episode_records(self, rewards: torch.Tensor) -> None:
        if self._episode_logger is None:
            return
        for packet in self.build_sensor_packets(rewards):
            state = packet.ground_truth
            assert state is not None
            index = int(packet.namespace.rsplit("_", maxsplit=1)[-1])
            localization = self._last_localization
            estimated_position = state.position_w if localization is None else localization.position_w[index]
            estimated_orientation = (
                state.orientation_wxyz if localization is None else localization.orientation_wxyz[index]
            )
            estimated_linear_velocity = (
                state.linear_velocity_w if localization is None else localization.linear_velocity_w[index]
            )
            estimated_angular_velocity = (
                state.angular_velocity_w
                if localization is None or localization.angular_velocity_w is None
                else localization.angular_velocity_w[index]
            )
            health = {
                "tracking_status": "tracking" if localization is None else localization.tracking_status,
                "feature_count": None
                if localization is None or localization.feature_count is None
                else localization.feature_count[index],
                "track_lifetime_s": None
                if localization is None or localization.track_lifetime_s is None
                else localization.track_lifetime_s[index],
                "spatial_coverage": None
                if localization is None or localization.spatial_coverage is None
                else localization.spatial_coverage[index],
                "innovation_norm": None
                if localization is None or localization.innovation_norm is None
                else localization.innovation_norm[index],
                "relocalization_event": False
                if localization is None or localization.relocalization_event is None
                else localization.relocalization_event[index],
            }
            self._episode_logger.write(
                {
                    "timestamp": packet.timestamp_s,
                    "gt_pose": {
                        "position_w": state.position_w,
                        "orientation_wxyz": state.orientation_wxyz,
                    },
                    "gt_velocity": {
                        "linear_velocity_w": state.linear_velocity_w,
                        "angular_velocity_w": state.angular_velocity_w,
                    },
                    "estimated_pose": {
                        "position_w": estimated_position,
                        "orientation_wxyz": estimated_orientation,
                    },
                    "estimated_velocity": {
                        "linear_velocity_w": estimated_linear_velocity,
                        "angular_velocity_w": estimated_angular_velocity,
                    },
                    "localization_health": health,
                    "control_command": packet.control_command,
                    "thruster_command": packet.thruster_command,
                    "sensor_metadata": packet.sensor_metadata,
                    "collision": packet.collision,
                    "goal": packet.goal_position_w,
                    "reward": packet.reward,
                    "namespace": packet.namespace,
                }
            )


@configclass
class UnderwaterPointNavPerceptionEnvCfg(UnderwaterPointNavEnvCfg):
    """Small headless perception configuration for RGB/depth/IMU validation."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=24.0,
        replicate_physics=True,
        # TiledCamera discovers USD prims, which Fabric-only clones do not expose.
        clone_in_fabric=False,
    )
    camera_sensor: TiledCameraCfg | None = SensorSuiteCfg(
        rgb_enabled=True,
        depth_enabled=True,
        imu_enabled=True,
        camera_width=160,
        camera_height=120,
        camera_rate_hz=15.0,
    ).camera_cfg("/World/envs/env_.*/Robot/Camera")
    imu_sensor: ImuCfg | None = SensorSuiteCfg(imu_enabled=True, imu_rate_hz=100.0).imu_cfg(
        "/World/envs/env_.*/Robot"
    )
    lighting: UnderwaterLightingCfg = UnderwaterLightingCfg(
        enabled=True,
        ambient_intensity=180.0,
        ambient_color_rgb=(0.08, 0.18, 0.22),
        vehicle_lights=(
            VehicleLightCfg(
                name="FrontLightLeft",
                position_b_m=(0.32, 0.12, 0.03),
                direction_b=(1.0, 0.0, 0.0),
                intensity=900.0,
            ),
            VehicleLightCfg(
                name="FrontLightRight",
                position_b_m=(0.32, -0.12, 0.03),
                direction_b=(1.0, 0.0, 0.0),
                intensity=900.0,
            ),
        ),
    )
    appearance: UnderwaterAppearanceCfg = UnderwaterAppearanceCfg(
        enabled=True,
        visibility_range_m=12.0,
        attenuation_rgb_per_m=(0.18, 0.07, 0.035),
        backscatter_strength=0.15,
    )


_STEREO_SENSOR_CFG = SensorSuiteCfg(
    rgb_enabled=True,
    depth_enabled=True,
    imu_enabled=True,
    camera_width=160,
    camera_height=120,
    camera_rate_hz=15.0,
    # The source-derived x=0.16 m camera points lie inside the opaque visual
    # hull (front bound is approximately x=0.222 m).  A near plane beyond the
    # hull suppresses self-occlusion until a transparent housing/CAD camera
    # mount is available.  This is an engineering guard, not calibration.
    camera_near_clip_m=0.25,
)

# Contact reporters are deliberately scoped to the visual cave pilot.  The
# open-water and legacy PointNav presets keep reporters disabled for throughput.
# GPU PhysX 5.1 cannot filter contacts against the imported triangle cave
# collider.  Keep the sensor unfiltered: the resulting force is a conservative
# static-collider contact signal (cave/seabed), not a cave-only ground truth.
_CAVE_CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot",
    update_period=0.0,
    track_air_time=False,
    debug_vis=False,
)


@configclass
class UnderwaterPointNavStereoEnvCfg(UnderwaterPointNavEnvCfg):
    """Stereo RGB/depth task with independent active-light commands.

    The policy still commands body velocity plus yaw rate.  The two extra
    action dimensions control left/right lamp intensity in ``[0, 1]`` after
    the normalized action mapping.  A future visual policy can therefore
    learn illumination scheduling without changing the vehicle controller.
    """

    action_space = 6
    light_control_enabled = True
    light_control_channels = 2
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=24.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )
    camera_sensor: TiledCameraCfg | None = None
    camera_left_sensor: TiledCameraCfg | None = _STEREO_SENSOR_CFG.stereo_camera_cfgs(
        "/World/envs/env_.*/Robot/CameraLeft",
        "/World/envs/env_.*/Robot/CameraRight",
    )[0]
    camera_right_sensor: TiledCameraCfg | None = _STEREO_SENSOR_CFG.stereo_camera_cfgs(
        "/World/envs/env_.*/Robot/CameraLeft",
        "/World/envs/env_.*/Robot/CameraRight",
    )[1]
    imu_sensor: ImuCfg | None = _STEREO_SENSOR_CFG.imu_cfg("/World/envs/env_.*/Robot")
    stereo_baseline_m = _STEREO_SENSOR_CFG.stereo_baseline_m
    stereo_near_clip_m = _STEREO_SENSOR_CFG.camera_near_clip_m
    lighting: UnderwaterLightingCfg = bluerov2_candidate_lighting(four_lights=False)
    appearance: UnderwaterAppearanceCfg = UnderwaterAppearanceCfg(
        enabled=True,
        visibility_range_m=12.0,
        attenuation_rgb_per_m=(0.18, 0.07, 0.035),
        backscatter_strength=0.15,
    )


@configclass
class UnderwaterPointNavImuEnvCfg(UnderwaterPointNavEnvCfg):
    """Camera-free IMU and vehicle-light runtime check."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8,
        env_spacing=24.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    camera_sensor: TiledCameraCfg | None = None
    imu_sensor: ImuCfg | None = SensorSuiteCfg(imu_enabled=True, imu_rate_hz=100.0).imu_cfg(
        "/World/envs/env_.*/Robot"
    )
    lighting: UnderwaterLightingCfg = UnderwaterLightingCfg(
        enabled=True,
        ambient_intensity=180.0,
        ambient_color_rgb=(0.08, 0.18, 0.22),
        vehicle_lights=(
            VehicleLightCfg(
                name="FrontLightLeft",
                position_b_m=(0.32, 0.12, 0.03),
                direction_b=(1.0, 0.0, 0.0),
                intensity=900.0,
            ),
            VehicleLightCfg(
                name="FrontLightRight",
                position_b_m=(0.32, -0.12, 0.03),
                direction_b=(1.0, 0.0, 0.0),
                intensity=900.0,
            ),
        ),
    )


def _default_cave_world(config_name: str, *, visual_enabled: bool) -> CaveWorldCfg:
    """Resolve the default cave registry without requiring conversion at import time."""
    world = load_cave_world_cfg(config_name, require_converted=False)
    world.visual_enabled = visual_enabled
    return world


def _default_multi_cave_world() -> CaveWorldCfg:
    """Resolve the selected v01 profile without requiring USDs at import time."""
    import os

    profile = os.environ.get("ISAAC_UNDERWATER_CAVE_PROFILE", "train_all")
    return load_cave_dataset_world_cfg(
        "worlds/caves_difficulty_v01.yaml",
        profile=profile,
        require_converted=False,
    )


def _multicave_domain_randomization_enabled() -> bool:
    import os

    return os.environ.get("ISAAC_UNDERWATER_MULTICAVE_DOMAIN_RANDOMIZATION", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@configclass
class UnderwaterCavePointNavEnvCfg(UnderwaterPointNavEnvCfg):
    """State/teacher cave task using collision-only cave copies for PPO throughput."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32,
        env_spacing=24.0,
        replicate_physics=False,
        # Fabric cloning currently fails on the imported triangle collision
        # mesh. Standard USD cloning is slower but keeps one cave per env.
        clone_in_fabric=False,
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=3,
        device="cuda:0",
        use_fabric=False,
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**18,
            gpu_max_rigid_patch_count=2**14,
            gpu_found_lost_pairs_capacity=2**16,
            gpu_found_lost_aggregate_pairs_capacity=2**16,
            gpu_total_aggregate_pairs_capacity=2**16,
            gpu_heap_capacity=2**24,
            gpu_temp_buffer_capacity=2**23,
        ),
    )
    world: CaveWorldCfg = _default_cave_world("worlds/synthetic_cave_turn90.yaml", visual_enabled=False)
    detailed_robot_visual = False


@configclass
class UnderwaterCavePerceptionEnvCfg(UnderwaterPointNavPerceptionEnvCfg):
    """Cave scene with visual mesh and RGB/depth/IMU sensors for perception smoke."""

    world: CaveWorldCfg = _default_cave_world("worlds/porth_yr_ogof_sump9.yaml", visual_enabled=True)


@configclass
class UnderwaterCaveStereoEnvCfg(UnderwaterPointNavStereoEnvCfg):
    """Porth cave scene with stereo RGB/depth and active-light control."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2,
        env_spacing=24.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )
    world: CaveWorldCfg = _default_cave_world("worlds/porth_yr_ogof_sump9.yaml", visual_enabled=True)
    detailed_robot_visual = True


@configclass
class UnderwaterCaveVisualPilotEnvCfg(UnderwaterCaveStereoEnvCfg):
    """Goal-conditioned visual actor with a privileged 19-D critic state.

    This is the first visual-PPO pilot.  The mission command is the local goal
    direction/distance already used by PointNav; it is not an entrance label or
    a cave geometry observation.  A later exploration task will replace this
    command with an explicit frontier/route objective.
    """

    observation_space = visual_feature_dim(
        output_hw=(12, 16),
        include_imu=True,
        include_pressure=True,
        include_previous_action=True,
        action_dim=6,
        mission_command_dim=4,
    )
    # navigation_observation_to_tensor emits 19 values; the legacy base cfg
    # predates the explicit three-axis angular velocity fields.
    state_space = 19
    visual_observation_enabled = True
    cave_route_reward_enabled = True
    robot: RigidObjectCfg = make_underwater_robot_cfg(
        load_config("robot.yaml"), activate_contact_sensors=True
    )
    contact_sensor: ContactSensorCfg | None = _CAVE_CONTACT_SENSOR_CFG
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2,
        env_spacing=24.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )


@configclass
class UnderwaterMultiCaveNavigationEnvCfg(UnderwaterCaveVisualPilotEnvCfg):
    """One recurrent exit-navigation policy trained across heterogeneous caves.

    Every environment is statically bound to one cave for the lifetime of the
    simulator.  The round-robin assignment balances transition counts while
    all environments update the same PPO actor/critic weights.
    """

    episode_length_s = 180.0
    goal_threshold_m = 1.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=6,
        # The hard mesh spans roughly 74 x 50 m.  Wide spacing prevents static
        # triangle meshes from adjacent environments overlapping in PhysX.
        env_spacing=200.0,
        replicate_physics=False,
        # Geometry cannot overlap at 200 m spacing, so an expensive global
        # collision collection is unnecessary for this heterogeneous task.
        filter_collisions=False,
        clone_in_fabric=False,
    )
    world: CaveWorldCfg = _default_multi_cave_world()
    domain_randomization_enabled = _multicave_domain_randomization_enabled()

    cave_route_reward_enabled = True
    cave_contact_termination_enabled = True
    cave_spawn_yaw_jitter_rad = 0.35
    route_progress_weight = 2.5
    route_deviation_penalty_weight = 0.35
    route_deviation_soft_m = 0.65
    route_deviation_hard_m = 2.25
    centerline_clearance_penalty_weight = 0.0
    collision_penalty = 25.0
    progress_weight = 0.25
    goal_bonus = 100.0
    heading_weight = 0.0
    action_penalty_weight = 0.002
    out_of_bounds_penalty = 25.0


@configclass
class UnderwaterCaveExploreEnvCfg(UnderwaterCaveVisualPilotEnvCfg):
    """Label-free cave coverage curriculum for the stereo CNN/GRU actor.

    The actor receives no goal vector, centerline coordinate, visitation map,
    or entrance label.  A fixed collision-checked, centerline-derived spawn is
    still used; the separate entry task owns portal inference and entry metrics.
    """

    episode_length_s = 30.0
    observation_space = visual_feature_dim(
        output_hw=(12, 16),
        include_imu=True,
        include_pressure=True,
        include_previous_action=True,
        action_dim=6,
        mission_command_dim=0,
    )
    state_space = 21
    visual_mission_command_enabled = False
    exploration_reward_enabled = True
    exploration_voxel_size_m = 0.5
    exploration_new_voxel_reward = 1.0
    exploration_revisit_penalty = 0.002
    exploration_clearance_margin_m = 0.8
    exploration_clearance_penalty_weight = 2.0
    exploration_random_yaw = True

    # Disable every goal/route shaping term.  Contact, action cost, workspace
    # bounds, visual clearance, and new-voxel reward remain active.
    progress_weight = 0.0
    goal_bonus = 0.0
    heading_weight = 0.0
    cave_route_reward_enabled = False


@configclass
class UnderwaterCaveEntryEnvCfg(UnderwaterCaveExploreEnvCfg):
    """Find and cross a geometry-inferred cave portal using vision only.

    Reset and success geometry comes from an automatic skeleton-clearance
    transition.  The actor observation remains exactly the same 1549-D
    no-command stereo contract as the coverage curriculum.
    """

    episode_length_s = 20.0
    state_space = 24
    cave_entry_enabled = True
    cave_entry_minimum_clearance_m = 0.65
    cave_entry_minimum_exterior_run_m = 1.0
    cave_entry_tangent_probe_m = 1.5
    cave_entry_spawn_distance_range_m = (1.5, 2.5)
    cave_entry_spawn_lateral_jitter_m = 0.10
    cave_entry_spawn_vertical_jitter_m = 0.05
    cave_entry_gate_radius_m = 1.5
    cave_entry_depth_m = 0.75
    cave_entry_bonus = 25.0
    cave_entry_terminate_on_success = True
    cave_contact_termination_enabled = True

    # Intrinsic novelty is only a weak search prior here; crossing the inferred
    # portal is the dominant sparse event.  No dense direction reward is used.
    exploration_new_voxel_reward = 0.05
    exploration_revisit_penalty = 0.001
