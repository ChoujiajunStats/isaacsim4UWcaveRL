#!/usr/bin/env python3
"""Standalone checks for GT/VIO separation and robot namespaces."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.interfaces import GroundTruthState, SensorPacket, TrackingStatus
from isaac_underwater.localization import ExternalVIOBackend, GroundTruthLocalizationBackend
from isaac_underwater.navigation import PolicyStateSource, build_navigation_observation
from isaac_underwater.sensors import robot_namespace


def main() -> None:
    zeros = torch.zeros(2, 3)
    state = GroundTruthState(
        timestamp_s=1.25,
        position_w=zeros.clone(),
        orientation_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        linear_velocity_w=zeros.clone(),
        angular_velocity_w=zeros.clone(),
    )
    packet = SensorPacket(namespace=robot_namespace(0), timestamp_s=1.25, ground_truth=state)
    estimated = GroundTruthLocalizationBackend().update(packet)
    assert estimated.tracking_status is TrackingStatus.TRACKING
    goal = torch.tensor([[1.0, 2.0, -1.0]]).repeat(2, 1)
    obs = build_navigation_observation(goal, PolicyStateSource.GROUND_TRUTH, state)
    assert torch.allclose(obs.goal_vector_b, goal)
    vio = ExternalVIOBackend()
    vio.submit(
        estimated.__class__(
            timestamp_s=1.25,
            position_w=state.position_w,
            orientation_wxyz=state.orientation_wxyz,
            linear_velocity_w=state.linear_velocity_w,
            angular_velocity_w=state.angular_velocity_w,
            covariance=torch.eye(6).repeat(2, 1, 1),
            tracking_status=TrackingStatus.TRACKING,
        )
    )
    global_batch = ExternalVIOBackend()
    global_batch.submit(estimated)
    assert global_batch.update_batch([packet, packet]).position_w.shape == (2, 3)
    vio_obs = build_navigation_observation(goal, PolicyStateSource.VIO, state, vio.update(packet))
    assert vio_obs.localization_status is TrackingStatus.TRACKING
    assert vio_obs.angular_velocity_b is not None
    vio.submit_for_robot("robot_001", estimated.__class__(
        timestamp_s=1.25,
        position_w=state.position_w[1] + 1.0,
        orientation_wxyz=state.orientation_wxyz[1],
        linear_velocity_w=state.linear_velocity_w[1],
        angular_velocity_w=state.angular_velocity_w[1],
        covariance=torch.eye(6),
        tracking_status=TrackingStatus.DEGRADED,
    ))
    vio.submit_for_robot("robot_000", estimated.__class__(
        timestamp_s=1.25,
        position_w=state.position_w[0],
        orientation_wxyz=state.orientation_wxyz[0],
        linear_velocity_w=state.linear_velocity_w[0],
        angular_velocity_w=state.angular_velocity_w[0],
        covariance=torch.eye(6),
        tracking_status=TrackingStatus.TRACKING,
    ))
    robot_packet = SensorPacket(namespace="robot_001", timestamp_s=1.25, ground_truth=state)
    assert vio.update(robot_packet).tracking_status is TrackingStatus.DEGRADED
    robot0_state = GroundTruthState(
        timestamp_s=state.timestamp_s,
        position_w=state.position_w[0],
        orientation_wxyz=state.orientation_wxyz[0],
        linear_velocity_w=state.linear_velocity_w[0],
        angular_velocity_w=state.angular_velocity_w[0],
    )
    batch = vio.update_batch(
        [
            SensorPacket(namespace="robot_000", timestamp_s=1.25, ground_truth=robot0_state),
            robot_packet,
        ]
    )
    assert batch.position_w.shape == (2, 3)
    assert batch.covariance.shape == (2, 6, 6)
    assert batch.tracking_status is TrackingStatus.DEGRADED
    assert packet.namespace == "robot_000"
    print("interfaces: PASS")


if __name__ == "__main__":
    main()
