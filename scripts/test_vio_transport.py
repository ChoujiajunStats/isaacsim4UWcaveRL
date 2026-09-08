#!/usr/bin/env python3
"""Check external-VIO serialization and UDP transport without Isaac Sim."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.interfaces import LocalizationOutput, SensorPacket, TrackingStatus
from isaac_underwater.localization import (
    ExternalVIOBackend,
    SensorJsonlExporter,
    UdpLocalizationReceiver,
    localization_output_from_dict,
    localization_output_to_dict,
    sensor_packet_to_dict,
)


def main() -> None:
    output = LocalizationOutput(
        timestamp_s=1.0,
        position_w=torch.tensor([1.0, 2.0, 3.0]),
        orientation_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        linear_velocity_w=torch.zeros(3),
        angular_velocity_w=torch.zeros(3),
        covariance=torch.eye(6),
        tracking_status=TrackingStatus.TRACKING,
        feature_count=torch.tensor(42.0),
    )
    payload = localization_output_to_dict(output)
    restored = localization_output_from_dict(payload)
    assert torch.allclose(restored.position_w, output.position_w)
    packet = SensorPacket(
        namespace="robot_000",
        timestamp_s=1.0,
        imu_acceleration=torch.zeros(3),
        imu_angular_velocity=torch.zeros(3),
    )
    assert sensor_packet_to_dict(packet)["namespace"] == "robot_000"
    path = PROJECT_ROOT / "logs" / "vio_transport_test.jsonl"
    if path.exists():
        path.unlink()
    SensorJsonlExporter(path).write(packet)
    assert json.loads(path.read_text().splitlines()[-1])["timestamp_s"] == 1.0

    backend = ExternalVIOBackend()
    receiver = UdpLocalizationReceiver("127.0.0.1", 0)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    payload["namespace"] = "robot_000"
    sender.sendto(json.dumps(payload).encode("utf-8"), receiver.address)
    assert receiver.poll(backend)
    assert backend.latest_for_robot("robot_000") is not None
    receiver.close()
    sender.close()
    print("vio_transport: PASS")


if __name__ == "__main__":
    main()
