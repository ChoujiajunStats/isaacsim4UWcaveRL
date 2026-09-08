"""Navigation observation construction independent of state source."""

from enum import Enum

import torch

from isaac_underwater.interfaces import GroundTruthState, LocalizationOutput, NavigationObservation, TrackingStatus
from isaac_underwater.physics import rotate_world_to_body


class PolicyStateSource(str, Enum):
    GROUND_TRUTH = "ground_truth"
    VIO = "vio"


def build_navigation_observation(
    goal_position_w: torch.Tensor,
    source: PolicyStateSource,
    ground_truth: GroundTruthState,
    localization: LocalizationOutput | None = None,
) -> NavigationObservation:
    if source is PolicyStateSource.GROUND_TRUTH:
        position = ground_truth.position_w
        orientation = ground_truth.orientation_wxyz
        velocity = ground_truth.linear_velocity_w
        covariance = torch.zeros(*position.shape[:-1], 6, 6, device=position.device)
        status = TrackingStatus.TRACKING
        timestamp = ground_truth.timestamp_s
    else:
        if localization is None:
            raise ValueError("VIO state source requires LocalizationOutput")
        position = localization.position_w
        orientation = localization.orientation_wxyz
        velocity = localization.linear_velocity_w
        covariance = localization.covariance
        status = localization.tracking_status
        timestamp = localization.timestamp_s
    goal_vector_b = rotate_world_to_body(orientation, goal_position_w - position)
    linear_velocity_b = rotate_world_to_body(orientation, velocity)
    angular_velocity_w = (
        ground_truth.angular_velocity_w
        if source is PolicyStateSource.GROUND_TRUTH
        else localization.angular_velocity_w
    )
    if angular_velocity_w is None:
        angular_velocity_w = torch.zeros_like(velocity)
    angular_velocity_b = rotate_world_to_body(orientation, angular_velocity_w)
    return NavigationObservation(
        timestamp_s=timestamp,
        goal_position_w=goal_position_w,
        position_w=position,
        orientation_wxyz=orientation,
        linear_velocity_w=velocity,
        goal_vector_b=goal_vector_b,
        localization_status=status,
        localization_covariance=covariance,
        linear_velocity_b=linear_velocity_b,
        angular_velocity_b=angular_velocity_b,
        feature_count=None if source is PolicyStateSource.GROUND_TRUTH else localization.feature_count,
        track_lifetime_s=None if source is PolicyStateSource.GROUND_TRUTH else localization.track_lifetime_s,
        spatial_coverage=None if source is PolicyStateSource.GROUND_TRUTH else localization.spatial_coverage,
        innovation_norm=None if source is PolicyStateSource.GROUND_TRUTH else localization.innovation_norm,
        relocalization_event=None
        if source is PolicyStateSource.GROUND_TRUTH
        else localization.relocalization_event,
    )


def navigation_observation_to_tensor(
    observation: NavigationObservation,
    projected_gravity_b: torch.Tensor,
    previous_action: torch.Tensor,
    *,
    position_scale: float = 0.1,
    linear_velocity_scale: float = 0.5,
    angular_velocity_scale: float = 0.5,
) -> torch.Tensor:
    """Create the compact PointNav policy tensor without exposing simulator objects."""
    distance = torch.linalg.vector_norm(observation.goal_vector_b, dim=-1, keepdim=True)
    if observation.linear_velocity_b is None or observation.angular_velocity_b is None:
        raise ValueError("NavigationObservation requires body-frame velocities for policy conversion")
    return torch.cat(
        (
            observation.goal_vector_b * position_scale,
            distance * position_scale,
            observation.linear_velocity_b * linear_velocity_scale,
            observation.angular_velocity_b * angular_velocity_scale,
            projected_gravity_b,
            previous_action,
        ),
        dim=-1,
    )
