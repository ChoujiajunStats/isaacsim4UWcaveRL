"""Load nominal/range/provenance records without hiding uncertainty."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from isaac_underwater.config import load_config


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    nominal: Any
    lower: Any
    upper: Any
    unit: str
    source_type: str
    source: str
    confidence: str
    notes: str = ""


def load_parameter_specs(filename: str = "robots/bluerov2_hydro.yaml") -> dict[str, ParameterSpec]:
    """Return flattened scalar/vector parameter records from the hydro YAML."""
    cfg = load_config(filename)
    specs: dict[str, ParameterSpec] = {}

    def visit(prefix: str, node: Any) -> None:
        if not isinstance(node, dict):
            return
        if {"nominal", "lower", "upper", "unit", "source_type", "source", "confidence"} <= node.keys():
            specs[prefix] = ParameterSpec(
                name=prefix,
                nominal=node["nominal"],
                lower=node["lower"],
                upper=node["upper"],
                unit=str(node["unit"]),
                source_type=str(node["source_type"]),
                source=str(node["source"]),
                confidence=str(node["confidence"]),
                notes=str(node.get("notes", "")),
            )
            return
        for key, value in node.items():
            if key in {"presets", "experimental_references"}:
                continue
            visit(f"{prefix}.{key}" if prefix else key, value)

    visit("", cfg)
    return specs
