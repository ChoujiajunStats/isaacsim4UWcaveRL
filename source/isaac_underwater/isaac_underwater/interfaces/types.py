"""Simulator-independent data contracts shared with future real robots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class ControlMode(str, Enum):
    BODY_VELOCITY = "body_velocity"
    BODY_WRENCH = "body_wrench"
    THRUSTER = "thruster"


class TrackingStatus(str, Enum):
    UNINITIALIZED = "uninitialized"
    TRACKING = "tracking"
    DEGRADED = "degraded"
    LOST = "lost"
    RELOCALIZED = "relocalized"


@dataclass(frozen=True)
class ControlCommand:
    timestamp_s: float
    body_velocity: torch.Tensor | None = None
    yaw_rate: torch.Tensor | None = None
    mode: ControlMode = ControlMode.BODY_VELOCITY
    body_wrench: torch.Tensor | None = None
    thruster_command: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Keep command mode and payload explicit at the simulator boundary."""
        if isinstance(self.mode, str):
            object.__setattr__(self, "mode", ControlMode(self.mode))
        if self.mode is ControlMode.BODY_VELOCITY and (self.body_velocity is None or self.yaw_rate is None):
            raise ValueError("BODY_VELOCITY commands require body_velocity and yaw_rate")
        if self.mode is ControlMode.BODY_WRENCH and self.body_wrench is None:
            raise ValueError("BODY_WRENCH commands require body_wrench")
        if self.mode is ControlMode.THRUSTER and self.thruster_command is None:
            raise ValueError("THRUSTER commands require thruster_command")

    @classmethod
    def from_body_velocity(
        cls, timestamp_s: float, body_velocity: torch.Tensor, yaw_rate: torch.Tensor
    ) -> "ControlCommand":
        return cls(timestamp_s, body_velocity, yaw_rate, ControlMode.BODY_VELOCITY)

    @classmethod
    def from_body_wrench(cls, timestamp_s: float, body_wrench: torch.Tensor) -> "ControlCommand":
        return cls(timestamp_s, mode=ControlMode.BODY_WRENCH, body_wrench=body_wrench)

    @classmethod
    def from_thrusters(cls, timestamp_s: float, thruster_command: torch.Tensor) -> "ControlCommand":
        return cls(timestamp_s, mode=ControlMode.THRUSTER, thruster_command=thruster_command)


@dataclass(frozen=True)
class GroundTruthState:
    timestamp_s: float
    position_w: torch.Tensor
    orientation_wxyz: torch.Tensor
    linear_velocity_w: torch.Tensor
    angular_velocity_w: torch.Tensor


@dataclass(frozen=True)
class LocalizationOutput:
    timestamp_s: float
    position_w: torch.Tensor
    orientation_wxyz: torch.Tensor
    linear_velocity_w: torch.Tensor
    covariance: torch.Tensor
    tracking_status: TrackingStatus
    angular_velocity_w: torch.Tensor | None = None
    feature_count: torch.Tensor | None = None
    track_lifetime_s: torch.Tensor | None = None
    spatial_coverage: torch.Tensor | None = None
    innovation_norm: torch.Tensor | None = None
    relocalization_event: torch.Tensor | None = None


@dataclass(frozen=True)
class SensorPacket:
    namespace: str
    timestamp_s: float
    rgb: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    # Stereo fields are optional so existing mono RGB/depth consumers remain
    # source-compatible.  A stereo packet carries both images in the same
    # timestamp domain as the legacy fields.
    rgb_left: torch.Tensor | None = None
    rgb_right: torch.Tensor | None = None
    depth_left: torch.Tensor | None = None
    depth_right: torch.Tensor | None = None
    imu_acceleration: torch.Tensor | None = None
    imu_angular_velocity: torch.Tensor | None = None
    pressure_depth_m: torch.Tensor | None = None
    auxiliary: dict[str, torch.Tensor] | None = None
    ground_truth: GroundTruthState | None = None
    control_command: ControlCommand | None = None
    thruster_command: torch.Tensor | None = None
    sensor_metadata: dict[str, object] | None = None
    collision: bool | torch.Tensor = False
    goal_position_w: torch.Tensor | None = None
    reward: torch.Tensor | float | None = None


@dataclass(frozen=True)
class NavigationObservation:
    timestamp_s: float
    goal_position_w: torch.Tensor
    position_w: torch.Tensor
    orientation_wxyz: torch.Tensor
    linear_velocity_w: torch.Tensor
    goal_vector_b: torch.Tensor
    localization_status: TrackingStatus
    localization_covariance: torch.Tensor
    linear_velocity_b: torch.Tensor | None = None
    angular_velocity_b: torch.Tensor | None = None
    feature_count: torch.Tensor | None = None
    track_lifetime_s: torch.Tensor | None = None
    spatial_coverage: torch.Tensor | None = None
    innovation_norm: torch.Tensor | None = None
    relocalization_event: torch.Tensor | None = None
