"""Low-VRAM vectorized underwater point-navigation environment."""

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import Imu, ImuCfg, TiledCamera, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaac_underwater.actuators import ThrusterAllocator, ThrusterModel, bluerov2_thruster_layout
from isaac_underwater.appearance import (
    UnderwaterAppearanceCfg,
    UnderwaterLightingCfg,
    VehicleLightCfg,
    apply_underwater_appearance,
    spawn_underwater_lighting,
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
    PolicyStateSource,
    build_navigation_observation,
    navigation_observation_to_tensor,
)
from isaac_underwater.physics import CurrentField, CurrentProfileCfg, Hydrodynamics, HydrodynamicsCfg, rotate_world_to_body
from isaac_underwater.randomization import (
    domain_gap_from_mapping,
    perturb_thruster_command,
    rotate_body_vectors_z,
    sample_domain_gap,
)
from isaac_underwater.robot import make_underwater_robot_cfg, spawn_bluerov2_visual
from isaac_underwater.sensors import SensorSuiteCfg
from isaac_underwater.worlds import OpenWaterWorldCfg, spawn_open_water_world


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
    imu_sensor: ImuCfg | None = None
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
        self._camera: TiledCamera | None = None
        self._imu: Imu | None = None
        self._last_control_command: ControlCommand | None = None
        self._last_thruster_command: torch.Tensor | None = None
        self._last_localization: LocalizationOutput | None = None
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._desired_velocity_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw_rate = torch.zeros(self.num_envs, device=self.device)
        self._goal_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_distance = torch.zeros(self.num_envs, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._out_of_bounds = torch.zeros_like(self._success)
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
            for key in ("progress", "goal", "heading", "action", "bounds")
        }

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
        if self.cfg.imu_sensor is not None:
            self._imu = Imu(self.cfg.imu_sensor)
        spawn_underwater_lighting(
            self.cfg.lighting,
            include_vehicle_lights=not self.cfg.scene.clone_in_fabric,
        )
        spawn_open_water_world(self.cfg.world, root_path="/World", seabed_path="/World/seabed")
        self.scene.clone_environments(copy_from_source=False)
        if self.cfg.lighting.enabled and self.cfg.scene.clone_in_fabric:
            for env_index in range(self.num_envs):
                spawn_underwater_lighting(
                    self.cfg.lighting,
                    root_path=f"/World/envs/env_{env_index}",
                    robot_path=f"/World/envs/env_{env_index}/Robot",
                    include_ambient=False,
                )
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/seabed"])
        self.scene.rigid_objects["robot"] = self._robot
        if self._camera is not None:
            self.scene.sensors["camera"] = self._camera
        if self._imu is not None:
            self.scene.sensors["imu"] = self._imu

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._desired_velocity_b = self._actions[:, :3] * self._max_command_velocity
        self._desired_yaw_rate = self._actions[:, 3] * self.cfg.max_command_yaw_rate_radps

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
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
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
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        total_reward = torch.stack(tuple(rewards.values())).sum(dim=0)
        self._write_episode_records(total_reward)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        relative_pos = self._robot.data.root_pos_w - self.scene.env_origins
        half_x = self.cfg.workspace_size_m[0] * 0.5
        half_y = self.cfg.workspace_size_m[1] * 0.5
        self._out_of_bounds = (
            (relative_pos[:, 0].abs() > half_x)
            | (relative_pos[:, 1].abs() > half_y)
            | (relative_pos[:, 2] < 0.25)
            | (relative_pos[:, 2] > self.cfg.workspace_size_m[2])
        )
        distance = torch.linalg.vector_norm(self._goal_pos_w - self._robot.data.root_pos_w, dim=-1)
        self._success = distance < self.cfg.goal_threshold_m
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self._out_of_bounds | self._success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

        if env_ids.numel() > 0:
            self.extras["log"] = {
                f"Episode_Reward/{key}": self._episode_sums[key][env_ids].mean().item()
                for key in self._episode_sums
            }
            self.extras["log"]["Metrics/success_rate"] = self._success[env_ids].float().mean().item()
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

        count = env_ids.numel()
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, :2] += torch.empty(count, 2, device=self.device).uniform_(
            -self.cfg.reset_xy_m, self.cfg.reset_xy_m
        )
        root_state[:, 2] = torch.empty(count, device=self.device).uniform_(*self.cfg.reset_depth_range_m)
        yaw = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        root_state[:, 3:7] = 0.0
        root_state[:, 3] = torch.cos(0.5 * yaw)
        root_state[:, 6] = torch.sin(0.5 * yaw)
        root_state[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        radius = torch.empty(count, device=self.device).uniform_(*self.cfg.goal_distance_range_m)
        self._goal_pos_w[env_ids, 0] = self.scene.env_origins[env_ids, 0] + radius * torch.cos(angle)
        self._goal_pos_w[env_ids, 1] = self.scene.env_origins[env_ids, 1] + radius * torch.sin(angle)
        vertical_offset = torch.empty(count, device=self.device).uniform_(
            -self.cfg.goal_vertical_offset_m, self.cfg.goal_vertical_offset_m
        )
        self._goal_pos_w[env_ids, 2] = (root_state[:, 2] + vertical_offset).clamp(
            0.5, self.cfg.workspace_size_m[2] - 0.5
        )
        self._previous_distance[env_ids] = torch.linalg.vector_norm(
            self._goal_pos_w[env_ids] - root_state[:, :3], dim=-1
        )
        self._success[env_ids] = False
        self._out_of_bounds[env_ids] = False

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
        if self._camera is not None:
            rgb = self._camera.data.output.get("rgb")
            depth = self._camera.data.output.get("depth")
            if rgb is not None and depth is not None:
                appearance_kwargs = {}
                if self._domain_gap_samples is not None:
                    appearance_kwargs = {
                        "visibility_range_m": self._domain_gap_samples["visibility_range_m"],
                        "attenuation_scale": self._domain_gap_samples["attenuation_scale"],
                        "backscatter_strength": self._domain_gap_samples["backscatter_strength"],
                        "exposure_offset": self._domain_gap_samples["exposure_offset"],
                        "motion_blur_strength": self._domain_gap_samples["motion_blur_strength"],
                    }
                rgb = apply_underwater_appearance(rgb, depth, self.cfg.appearance, **appearance_kwargs)
            if self._domain_gap_samples is not None and rgb is not None:
                rgb_noise = torch.randn(rgb.shape, device=self.device)
                rgb_std = self._domain_gap_samples["camera_noise_std"].reshape(
                    self.num_envs, *([1] * (rgb.ndim - 1))
                )
                if rgb.dtype == torch.uint8:
                    rgb = (rgb.float() + rgb_noise * rgb_std * 255.0).clamp(0.0, 255.0).to(torch.uint8)
                else:
                    rgb = (rgb.float() + rgb_noise * rgb_std).clamp(0.0, 1.0).to(rgb.dtype)
            if self._domain_gap_samples is not None and depth is not None:
                depth_noise = torch.randn(depth.shape, device=self.device)
                depth_std = self._domain_gap_samples["depth_noise_std"].reshape(
                    self.num_envs, *([1] * (depth.ndim - 1))
                )
                depth = (depth.float() + depth_noise * depth_std).clamp_min(0.0).to(depth.dtype)
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
                    rgb=None if rgb is None or dropped else rgb[robot_index],
                    depth=None if depth is None or dropped else depth[robot_index],
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
                        "imu": imu_acceleration is not None,
                        "pressure": True,
                        "ground_truth": True,
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
                    collision=bool(self._out_of_bounds[robot_index].item()),
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
