#!/usr/bin/env python3
"""Example external localization adapter for JSONL sensor packets.

This validates the process/transport contract only. It is an IMU dead
reckoner, not a replacement for OpenVINS or another camera-based VIO system.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import torch

from isaac_underwater.localization import (
    ImuDeadReckoningBackend,
    localization_output_to_dict,
    sensor_packet_from_dict,
)


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True, help="JSONL SensorPacket stream")
parser.add_argument("--output", type=Path, default=None, help="optional JSONL LocalizationOutput stream")
parser.add_argument("--udp-host", type=str, default=None)
parser.add_argument("--udp-port", type=int, default=None)
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()


def main() -> None:
    if (args.udp_host is None) != (args.udp_port is None):
        raise ValueError("--udp-host and --udp-port must be provided together")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp_host is not None else None
    backend = ImuDeadReckoningBackend()
    count = 0
    try:
        output_stream = args.output.open("w", encoding="utf-8") if args.output is not None else None
        try:
            with args.input.open(encoding="utf-8") as input_stream:
                for line in input_stream:
                    if not line.strip():
                        continue
                    packet = sensor_packet_from_dict(json.loads(line), device=args.device)
                    output = localization_output_to_dict(backend.update(packet))
                    output["namespace"] = packet.namespace
                    encoded = json.dumps(output, separators=(",", ":"))
                    if output_stream is not None:
                        output_stream.write(encoded + "\n")
                    if sender is not None:
                        sender.sendto(encoded.encode("utf-8"), (args.udp_host, args.udp_port))
                    count += 1
        finally:
            if output_stream is not None:
                output_stream.close()
    finally:
        if sender is not None:
            sender.close()
    print(f"processed_packets={count} status=degraded adapter=imu_dead_reckoning")


if __name__ == "__main__":
    main()
