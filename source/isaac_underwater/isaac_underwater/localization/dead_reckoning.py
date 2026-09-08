"""Small external-process adapter used to validate the localization boundary.

This is intentionally not a visual-inertial odometry implementation.  It
integrates the IMU stream, reports degraded health, and provides a deterministic
stand-in while OpenVINS/ORB-SLAM is developed behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaac_underwater.interfaces import LocalizationOutput, SensorPacket, TrackingStatus


def _normalize(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(quaternion).clamp_min(1.0e-8)


def _quaternion_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind()
    w2, x2, y2, z2 = second.unbind()
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _rotate_body_to_world(quaternion: torch.Tensor, vector_b: torch.Tensor) -> torch.Tensor:
    conjugate = quaternion * torch.tensor(
        (1.0, -1.0, -1.0, -1.0), device=quaternion.device, dtype=quaternion.dtype
    )
    pure = torch.cat((torch.zeros(1, device=vector_b.device, dtype=vector_b.dtype), vector_b))
    return _quaternion_multiply(_quaternion_multiply(quaternion, pure), conjugate)[1:]


def _integrate_orientation(
    quaternion: torch.Tensor, angular_velocity_b: torch.Tensor, dt: float
) -> torch.Tensor:
    angle = torch.linalg.vector_norm(angular_velocity_b) * dt
    half_angle = 0.5 * angle
    axis = angular_velocity_b / torch.linalg.vector_norm(angular_velocity_b).clamp_min(1.0e-8)
    delta = torch.cat((half_angle.cos().reshape(1), axis * half_angle.sin()))
    return _normalize(_quaternion_multiply(quaternion, delta))


@dataclass
class _DeadReckoningState:
    timestamp_s: float
    position_w: torch.Tensor
    orientation_wxyz: torch.Tensor
    linear_velocity_w: torch.Tensor
    track_lifetime_s: float


class ImuDeadReckoningBackend:
    """Dependency-free external adapter with explicit degraded health."""

    def __init__(self) -> None:
        self._states: dict[str, _DeadReckoningState] = {}

    def reset(self, namespace: str | None = None) -> None:
        if namespace is None:
            self._states.clear()
        else:
            self._states.pop(namespace, None)

    def update(self, packet: SensorPacket) -> LocalizationOutput:
        if packet.imu_acceleration is None or packet.imu_angular_velocity is None:
            raise ValueError("ImuDeadReckoningBackend requires acceleration and angular velocity")
        acceleration_b = packet.imu_acceleration.float()
        angular_velocity_b = packet.imu_angular_velocity.float()
        if acceleration_b.shape != (3,) or angular_velocity_b.shape != (3,):
            raise ValueError("dead-reckoning IMU vectors must have shape (3,)")

        state = self._states.get(packet.namespace)
        if state is None:
            state = _DeadReckoningState(
                timestamp_s=packet.timestamp_s,
                position_w=torch.zeros(3, device=acceleration_b.device, dtype=acceleration_b.dtype),
                orientation_wxyz=torch.tensor(
                    (1.0, 0.0, 0.0, 0.0), device=acceleration_b.device, dtype=acceleration_b.dtype
                ),
                linear_velocity_w=torch.zeros(3, device=acceleration_b.device, dtype=acceleration_b.dtype),
                track_lifetime_s=0.0,
            )
            self._states[packet.namespace] = state
        dt = packet.timestamp_s - state.timestamp_s
        if dt < 0.0:
            raise ValueError("sensor timestamps must be monotonic per namespace")
        if dt > 0.0:
            acceleration_w = _rotate_body_to_world(state.orientation_wxyz, acceleration_b)
            state.position_w = state.position_w + state.linear_velocity_w * dt + 0.5 * acceleration_w * (dt * dt)
            state.linear_velocity_w = state.linear_velocity_w + acceleration_w * dt
            state.orientation_wxyz = _integrate_orientation(state.orientation_wxyz, angular_velocity_b, dt)
            state.track_lifetime_s += dt
            state.timestamp_s = packet.timestamp_s

        uncertainty = 0.05 + 0.02 * state.track_lifetime_s
        covariance = torch.eye(6, device=state.position_w.device, dtype=state.position_w.dtype) * uncertainty
        angular_velocity_w = _rotate_body_to_world(state.orientation_wxyz, angular_velocity_b)
        return LocalizationOutput(
            timestamp_s=packet.timestamp_s,
            position_w=state.position_w.clone(),
            orientation_wxyz=state.orientation_wxyz.clone(),
            linear_velocity_w=state.linear_velocity_w.clone(),
            covariance=covariance,
            tracking_status=TrackingStatus.DEGRADED,
            angular_velocity_w=angular_velocity_w,
            feature_count=torch.tensor(0.0, device=state.position_w.device),
            track_lifetime_s=torch.tensor(state.track_lifetime_s, device=state.position_w.device),
            spatial_coverage=torch.tensor(0.0, device=state.position_w.device),
            innovation_norm=torch.tensor(0.0, device=state.position_w.device),
            relocalization_event=torch.tensor(False, device=state.position_w.device),
        )
