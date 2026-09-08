"""Small dependency-free transport helpers for an external VIO process."""

from __future__ import annotations

import json
import socket
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

from isaac_underwater.interfaces import LocalizationOutput, SensorPacket, TrackingStatus


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def sensor_packet_to_dict(packet: SensorPacket, *, include_images: bool = False) -> dict[str, Any]:
    """Serialize synchronized metadata for a subprocess or socket transport."""
    payload: dict[str, Any] = {
        "namespace": packet.namespace,
        "timestamp_s": packet.timestamp_s,
        "pressure_depth_m": _plain(packet.pressure_depth_m),
        "auxiliary": _plain(packet.auxiliary),
        "sensor_metadata": packet.sensor_metadata or {},
    }
    if include_images:
        payload["rgb"] = _plain(packet.rgb)
        payload["depth"] = _plain(packet.depth)
        payload["rgb_left"] = _plain(packet.rgb_left)
        payload["rgb_right"] = _plain(packet.rgb_right)
        payload["depth_left"] = _plain(packet.depth_left)
        payload["depth_right"] = _plain(packet.depth_right)
    if packet.imu_acceleration is not None:
        payload["imu_acceleration"] = _plain(packet.imu_acceleration)
    if packet.imu_angular_velocity is not None:
        payload["imu_angular_velocity"] = _plain(packet.imu_angular_velocity)
    return payload


def sensor_packet_from_dict(
    payload: dict[str, Any], device: torch.device | str = "cpu"
) -> SensorPacket:
    """Deserialize the transport subset consumed by an external estimator."""
    tensor = lambda value: None if value is None else torch.as_tensor(value, dtype=torch.float32, device=device)
    auxiliary = payload.get("auxiliary")
    if auxiliary is not None:
        auxiliary = {str(key): tensor(value) for key, value in auxiliary.items()}
    return SensorPacket(
        namespace=str(payload["namespace"]),
        timestamp_s=float(payload["timestamp_s"]),
        rgb=tensor(payload.get("rgb")),
        depth=tensor(payload.get("depth")),
        rgb_left=tensor(payload.get("rgb_left")),
        rgb_right=tensor(payload.get("rgb_right")),
        depth_left=tensor(payload.get("depth_left")),
        depth_right=tensor(payload.get("depth_right")),
        imu_acceleration=tensor(payload.get("imu_acceleration")),
        imu_angular_velocity=tensor(payload.get("imu_angular_velocity")),
        pressure_depth_m=tensor(payload.get("pressure_depth_m")),
        auxiliary=auxiliary,
        sensor_metadata=payload.get("sensor_metadata") or {},
    )


def localization_output_to_dict(output: LocalizationOutput) -> dict[str, Any]:
    return _plain(output)


def localization_output_from_dict(payload: dict[str, Any], device: torch.device | str = "cpu") -> LocalizationOutput:
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    return LocalizationOutput(
        timestamp_s=float(payload["timestamp_s"]),
        position_w=tensor(payload["position_w"]),
        orientation_wxyz=tensor(payload["orientation_wxyz"]),
        linear_velocity_w=tensor(payload["linear_velocity_w"]),
        covariance=tensor(payload["covariance"]),
        tracking_status=TrackingStatus(payload["tracking_status"]),
        angular_velocity_w=None
        if payload.get("angular_velocity_w") is None
        else tensor(payload["angular_velocity_w"]),
        feature_count=None if payload.get("feature_count") is None else tensor(payload["feature_count"]),
        track_lifetime_s=None
        if payload.get("track_lifetime_s") is None
        else tensor(payload["track_lifetime_s"]),
        spatial_coverage=None
        if payload.get("spatial_coverage") is None
        else tensor(payload["spatial_coverage"]),
        innovation_norm=None
        if payload.get("innovation_norm") is None
        else tensor(payload["innovation_norm"]),
        relocalization_event=None
        if payload.get("relocalization_event") is None
        else tensor(payload["relocalization_event"]),
    )


class SensorJsonlExporter:
    """Append sensor packets for an external VIO process to consume."""

    def __init__(self, path: str | Path, *, include_images: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_images = include_images

    def write(self, packet: SensorPacket) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            json.dump(sensor_packet_to_dict(packet, include_images=self.include_images), stream, separators=(",", ":"))
            stream.write("\n")


class UdpLocalizationReceiver:
    """Non-blocking UDP receiver that feeds JSON VIO outputs to a backend."""

    def __init__(self, host: str, port: int, *, device: torch.device | str = "cpu") -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        self.device = device

    @property
    def address(self) -> tuple[str, int]:
        return self.socket.getsockname()

    def poll(self, backend: Any) -> bool:
        try:
            payload, _ = self.socket.recvfrom(1 << 20)
        except BlockingIOError:
            return False
        message = json.loads(payload.decode("utf-8"))
        output = localization_output_from_dict(message, self.device)
        namespace = message.get("namespace")
        if namespace is not None and hasattr(backend, "submit_for_robot"):
            backend.submit_for_robot(str(namespace), output)
        else:
            backend.submit(output)
        return True

    def close(self) -> None:
        self.socket.close()
