"""Unified episode logging for simulation and dataset export."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def _serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "value"):
        return value.value
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


class EpisodeJsonlLogger:
    REQUIRED_FIELDS = {
        "timestamp",
        "gt_pose",
        "gt_velocity",
        "estimated_pose",
        "estimated_velocity",
        "localization_health",
        "control_command",
        "thruster_command",
        "sensor_metadata",
        "collision",
        "goal",
        "reward",
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        missing = self.REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"Missing episode log fields: {sorted(missing)}")
        payload = {key: _serializable(value) for key, value in record.items()}
        with self.path.open("a", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.write("\n")

    def write_many(self, records: list[dict[str, Any]]) -> None:
        """Append a batch while preserving the same schema for every robot."""
        for record in records:
            self.write(record)
