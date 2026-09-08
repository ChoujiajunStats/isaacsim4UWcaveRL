"""Replaceable localization backends with explicit health state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from isaac_underwater.interfaces import LocalizationOutput, SensorPacket, TrackingStatus


class LocalizationBackend(Protocol):
    def update(self, packet: SensorPacket) -> LocalizationOutput:
        """Consume a synchronized sensor packet and return deployable state."""


def stack_localization_outputs(outputs: Sequence[LocalizationOutput]) -> LocalizationOutput:
    """Combine independent robot estimates into one vectorized output."""
    if not outputs:
        raise ValueError("at least one localization output is required")
    statuses = [output.tracking_status for output in outputs]
    if all(status is statuses[0] for status in statuses):
        status = statuses[0]
    elif TrackingStatus.LOST in statuses:
        status = TrackingStatus.LOST
    elif TrackingStatus.DEGRADED in statuses:
        status = TrackingStatus.DEGRADED
    elif TrackingStatus.RELOCALIZED in statuses:
        status = TrackingStatus.RELOCALIZED
    else:
        status = TrackingStatus.TRACKING

    def stack_optional(name: str) -> torch.Tensor | None:
        values = [getattr(output, name) for output in outputs]
        if any(value is None for value in values):
            return None
        return torch.stack([value for value in values if value is not None])

    return LocalizationOutput(
        timestamp_s=max(output.timestamp_s for output in outputs),
        position_w=torch.stack([output.position_w for output in outputs]),
        orientation_wxyz=torch.stack([output.orientation_wxyz for output in outputs]),
        linear_velocity_w=torch.stack([output.linear_velocity_w for output in outputs]),
        covariance=torch.stack([output.covariance for output in outputs]),
        tracking_status=status,
        angular_velocity_w=stack_optional("angular_velocity_w"),
        feature_count=stack_optional("feature_count"),
        track_lifetime_s=stack_optional("track_lifetime_s"),
        spatial_coverage=stack_optional("spatial_coverage"),
        innovation_norm=stack_optional("innovation_norm"),
        relocalization_event=stack_optional("relocalization_event"),
    )


class GroundTruthLocalizationBackend:
    def update(self, packet: SensorPacket) -> LocalizationOutput:
        if packet.ground_truth is None:
            raise ValueError("GroundTruthLocalizationBackend requires ground_truth in SensorPacket")
        state = packet.ground_truth
        batch_shape = state.position_w.shape[:-1]
        covariance = torch.zeros(*batch_shape, 6, 6, device=state.position_w.device)
        return LocalizationOutput(
            timestamp_s=state.timestamp_s,
            position_w=state.position_w,
            orientation_wxyz=state.orientation_wxyz,
            linear_velocity_w=state.linear_velocity_w,
            covariance=covariance,
            tracking_status=TrackingStatus.TRACKING,
            angular_velocity_w=state.angular_velocity_w,
        )


class ExternalVIOBackend:
    """IPC-facing backend with optional per-robot output streams.

    A single output remains valid for a one-robot experiment.  Vectorized
    experiments can submit independent estimates keyed by ``SensorPacket``
    namespace without changing the navigation API.
    """

    def __init__(self) -> None:
        self._latest: LocalizationOutput | None = None
        self._latest_by_namespace: dict[str, LocalizationOutput] = {}

    def submit(self, output: LocalizationOutput, namespace: str | None = None) -> None:
        previous = self._latest_by_namespace.get(namespace) if namespace is not None else self._latest
        if previous is not None and output.timestamp_s < previous.timestamp_s:
            raise ValueError("Localization timestamps must be monotonic")
        if namespace is None:
            self._latest = output
        else:
            self._latest_by_namespace[namespace] = output

    def submit_for_robot(self, namespace: str, output: LocalizationOutput) -> None:
        self.submit(output, namespace=namespace)

    def update_batch(self, packets: Sequence[SensorPacket]) -> LocalizationOutput:
        """Update one estimate per packet and return a vectorized result."""
        if not packets:
            raise ValueError("at least one sensor packet is required")
        if not self._latest_by_namespace and self._latest is not None:
            if self._latest.timestamp_s > max(packet.timestamp_s for packet in packets):
                raise ValueError("ExternalVIO output is newer than the sensor packet")
            if len(packets) == 1:
                return self._latest
            if self._latest.position_w.ndim > 1:
                if self._latest.position_w.shape[0] != len(packets):
                    raise ValueError("batched ExternalVIO output does not match packet count")
                return self._latest
        return stack_localization_outputs([self.update(packet) for packet in packets])

    @property
    def latest(self) -> LocalizationOutput | None:
        return self._latest

    def latest_for_robot(self, namespace: str) -> LocalizationOutput | None:
        return self._latest_by_namespace.get(namespace, self._latest)

    def update(self, packet: SensorPacket) -> LocalizationOutput:
        latest = self.latest_for_robot(packet.namespace)
        if latest is None:
            raise RuntimeError("No ExternalVIO output has been submitted")
        if latest.timestamp_s > packet.timestamp_s:
            raise ValueError("ExternalVIO output is newer than the sensor packet")
        return latest
