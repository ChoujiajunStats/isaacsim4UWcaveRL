#!/usr/bin/env python3
"""Check the runnable external localization adapter without Isaac Sim."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))

from isaac_underwater.interfaces import SensorPacket, TrackingStatus
from isaac_underwater.localization import (
    ImuDeadReckoningBackend,
    SensorJsonlExporter,
    sensor_packet_from_dict,
    sensor_packet_to_dict,
)


def main() -> None:
    packets = [
        SensorPacket(
            namespace="robot_000",
            timestamp_s=timestamp,
            imu_acceleration=torch.tensor([1.0, 0.0, 0.0]),
            imu_angular_velocity=torch.zeros(3),
        )
        for timestamp in (0.0, 0.1, 0.2)
    ]
    backend = ImuDeadReckoningBackend()
    outputs = [backend.update(packet) for packet in packets]
    assert outputs[-1].tracking_status is TrackingStatus.DEGRADED
    assert outputs[-1].position_w[0] > 0.0
    assert outputs[-1].track_lifetime_s is not None
    assert torch.isfinite(outputs[-1].covariance).all()
    restored = sensor_packet_from_dict(sensor_packet_to_dict(packets[1]))
    assert restored.namespace == packets[1].namespace
    assert torch.allclose(restored.imu_acceleration, packets[1].imu_acceleration)

    with tempfile.TemporaryDirectory() as directory:
        input_path = Path(directory) / "sensors.jsonl"
        output_path = Path(directory) / "localization.jsonl"
        exporter = SensorJsonlExporter(input_path)
        for packet in packets:
            exporter.write(packet)
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "vio_dead_reckoning.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "processed_packets=3" in result.stdout
        rows = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert len(rows) == 3
        assert rows[-1]["namespace"] == "robot_000"
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(2.0)
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "vio_dead_reckoning.py"),
                    "--input",
                    str(input_path),
                    "--udp-host",
                    "127.0.0.1",
                    "--udp-port",
                    str(receiver.getsockname()[1]),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            datagram, _ = receiver.recvfrom(1 << 16)
            assert json.loads(datagram)["namespace"] == "robot_000"
        finally:
            receiver.close()
    print("vio_dead_reckoning: PASS")


if __name__ == "__main__":
    main()
