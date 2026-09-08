"""Validated cave-route loading independent of Isaac Sim.

The historical cave assets use CSV centerlines while the difficulty dataset
ships explicit JSON entrance-to-exit paths.  This module gives both formats a
single, small contract so task code does not need format-specific branches.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class CaveRoute:
    """A route in the same local Z-up frame as its source cave mesh."""

    points_m: tuple[Point3, ...]
    chainage_m: tuple[float, ...]
    start_m: Point3
    goal_m: Point3
    nominal_surface_clearance_m: tuple[float, ...] | None = None

    @property
    def length_m(self) -> float:
        return self.chainage_m[-1] - self.chainage_m[0]


def balanced_scene_assignment(num_envs: int, num_scenes: int) -> tuple[int, ...]:
    """Return a deterministic round-robin assignment with every scene present."""
    if num_scenes <= 0:
        raise ValueError("num_scenes must be positive")
    if num_envs < num_scenes:
        raise ValueError(
            f"num_envs ({num_envs}) must be at least the selected scene count ({num_scenes})"
        )
    return tuple(index % num_scenes for index in range(num_envs))


def _point3(value: Any, *, field: str, path: Path) -> Point3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{field} must contain three coordinates: {path}")
    point = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in point):
        raise ValueError(f"{field} contains a non-finite coordinate: {path}")
    return point  # type: ignore[return-value]


def _validate_route(
    points: tuple[Point3, ...],
    chainage: tuple[float, ...],
    *,
    path: Path,
) -> None:
    if len(points) < 2:
        raise ValueError(f"Cave route needs at least two points: {path}")
    if len(points) != len(chainage):
        raise ValueError(f"Cave route point/chainage lengths differ: {path}")
    if not all(math.isfinite(value) for value in chainage):
        raise ValueError(f"Cave route contains non-finite chainage: {path}")
    if any(right <= left for left, right in zip(chainage, chainage[1:])):
        raise ValueError(f"Cave route chainage must be strictly increasing: {path}")


def _json_route(path: Path) -> CaveRoute:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Cave navigation JSON must be an object: {path}")
    raw_points = data.get("points")
    if not isinstance(raw_points, list):
        raise ValueError(f"Cave navigation JSON has no points array: {path}")
    points = tuple(_point3(value, field=f"points[{index}]", path=path) for index, value in enumerate(raw_points))
    chainage_values = [0.0]
    for left, right in zip(points, points[1:]):
        step = math.dist(left, right)
        if step <= 1.0e-8:
            raise ValueError(f"Cave route contains duplicate consecutive points: {path}")
        chainage_values.append(chainage_values[-1] + step)
    chainage = tuple(chainage_values)
    _validate_route(points, chainage, path=path)
    start = _point3(data.get("start", points[0]), field="start", path=path)
    goal = _point3(data.get("goal", points[-1]), field="goal", path=path)
    return CaveRoute(points, chainage, start, goal)


def _csv_route(path: Path) -> CaveRoute:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = ("chainage_m", "north_m", "east_m", "down_m")
    if not rows or any(field not in rows[0] for field in required):
        raise ValueError(f"Cave centerline must contain columns {required}: {path}")
    points = tuple(
        (float(row["north_m"]), float(row["east_m"]), float(row["down_m"])) for row in rows
    )
    chainage = tuple(float(row["chainage_m"]) for row in rows)
    _validate_route(points, chainage, path=path)
    clearance = None
    if "nominal_surface_clearance_m" in rows[0]:
        clearance = tuple(float(row["nominal_surface_clearance_m"]) for row in rows)
    return CaveRoute(points, chainage, points[0], points[-1], clearance)


def load_cave_route(path: str | Path) -> CaveRoute:
    """Load and validate a JSON or CSV cave route."""
    route_path = Path(path).expanduser().resolve()
    if not route_path.is_file():
        raise FileNotFoundError(route_path)
    if route_path.suffix.lower() == ".json":
        return _json_route(route_path)
    if route_path.suffix.lower() == ".csv":
        return _csv_route(route_path)
    raise ValueError(f"Unsupported cave route format {route_path.suffix!r}: {route_path}")
