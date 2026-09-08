"""Underwater lighting knobs and an optional Isaac Lab spawner."""

from dataclasses import dataclass
import math
from collections.abc import Sequence
from typing import Any


@dataclass
class VehicleLightCfg:
    name: str
    position_b_m: tuple[float, float, float]
    direction_b: tuple[float, float, float]
    intensity: float = 1500.0
    beam_angle_deg: float = 135.0
    color_rgb: tuple[float, float, float] = (0.82, 0.9, 1.0)
    source_type: str = "estimated"
    source: str = ""
    confidence: str = "low"
    notes: str = ""


@dataclass
class UnderwaterLightingCfg:
    enabled: bool = False
    ambient_intensity: float = 250.0
    ambient_color_rgb: tuple[float, float, float] = (0.08, 0.18, 0.22)
    distance_attenuation: float = 0.12
    flicker_amplitude: float = 0.0
    vehicle_lights: tuple[VehicleLightCfg, ...] = ()


@dataclass
class UnderwaterAppearanceCfg:
    enabled: bool = False
    water_color_rgb: tuple[float, float, float] = (0.04, 0.18, 0.22)
    visibility_range_m: float = 12.0
    fog_density: float = 0.04
    attenuation_rgb_per_m: tuple[float, float, float] = (0.18, 0.07, 0.035)
    background_intensity: float = 0.2
    contrast: float = 0.9
    camera_exposure: float = 0.0
    backscatter_strength: float = 0.0


def coerce_lighting_cfg(value: UnderwaterLightingCfg | dict[str, Any]) -> UnderwaterLightingCfg:
    """Normalize direct and Hydra-loaded lighting configs to dataclass objects."""
    if isinstance(value, UnderwaterLightingCfg):
        data: dict[str, Any] = {
            "enabled": value.enabled,
            "ambient_intensity": value.ambient_intensity,
            "ambient_color_rgb": tuple(value.ambient_color_rgb),
            "distance_attenuation": value.distance_attenuation,
            "flicker_amplitude": value.flicker_amplitude,
            "vehicle_lights": value.vehicle_lights,
        }
    elif isinstance(value, dict):
        data = dict(value)
    else:
        raise TypeError(f"Expected UnderwaterLightingCfg or mapping, got {type(value)}")

    normalized_lights = []
    for light in data.get("vehicle_lights", ()):
        if isinstance(light, VehicleLightCfg):
            normalized_lights.append(light)
        elif isinstance(light, dict):
            normalized_lights.append(
                VehicleLightCfg(
                    name=str(light["name"]),
                    position_b_m=tuple(float(x) for x in light["position_b_m"]),
                    direction_b=tuple(float(x) for x in light["direction_b"]),
                    intensity=float(light.get("intensity", light.get("luminous_flux_lm", 1500.0))),
                    beam_angle_deg=float(light.get("beam_angle_deg", 135.0)),
                    color_rgb=tuple(float(x) for x in light.get("color_rgb", (0.82, 0.9, 1.0))),
                    source_type=str(light.get("source_type", "estimated")),
                    source=str(light.get("source", "")),
                    confidence=str(light.get("confidence", "low")),
                    notes=str(light.get("notes", "")),
                )
            )
        else:
            raise TypeError(f"Expected VehicleLightCfg or mapping, got {type(light)}")
    data["vehicle_lights"] = tuple(normalized_lights)
    return UnderwaterLightingCfg(**data)


def lighting_intensity(
    base_intensity: float,
    cfg: UnderwaterLightingCfg,
    time_s: float = 0.0,
    distance_m: float = 0.0,
) -> float:
    """Return deterministic temporal and distance-attenuated intensity."""
    amplitude = max(0.0, min(1.0, cfg.flicker_amplitude))
    temporal = 1.0 + amplitude * math.sin(2.0 * math.pi * time_s)
    distance = max(0.0, distance_m)
    attenuation = 1.0 / (1.0 + max(0.0, cfg.distance_attenuation) * distance * distance)
    return max(0.0, base_intensity * temporal * attenuation)


def _direction_to_quaternion(direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Rotate a light's local -Z axis toward a body-frame direction."""
    dx, dy, dz = direction
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 1.0e-8:
        raise ValueError("light direction must be non-zero")
    target = (dx / norm, dy / norm, dz / norm)
    source = (0.0, 0.0, -1.0)
    cross = (
        source[1] * target[2] - source[2] * target[1],
        source[2] * target[0] - source[0] * target[2],
        source[0] * target[1] - source[1] * target[0],
    )
    dot = sum(a * b for a, b in zip(source, target))
    if dot < -1.0 + 1.0e-6:
        return (0.0, 1.0, 0.0, 0.0)
    w = math.sqrt((1.0 + dot) * 0.5)
    scale = 0.5 / max(w, 1.0e-8)
    return (w, cross[0] * scale, cross[1] * scale, cross[2] * scale)


def _light_intensity_attribute(prim: Any) -> Any:
    """Resolve Isaac/Usd light intensity across schema versions."""
    if not hasattr(prim, "GetAttribute"):
        return None
    for name in ("inputs:intensity", "intensity"):
        attribute = prim.GetAttribute(name)
        if attribute is not None and attribute.IsValid():
            return attribute
    return None


def spawn_underwater_lighting(
    cfg: UnderwaterLightingCfg,
    *,
    root_path: str = "/World",
    robot_path: str = "/World/envs/env_0/Robot",
    include_ambient: bool = True,
    include_vehicle_lights: bool = True,
) -> list[Any]:
    """Spawn lightweight dome and vehicle lights and return their USD prims.

    The robot lights are children of the robot prim, so they inherit its pose
    when environments are cloned.  This function is intentionally opt-in and
    should not be called by physics-only tasks.
    """
    if not cfg.enabled:
        return []
    from isaaclab import sim as sim_utils

    prims = []
    if include_ambient:
        dome_cfg = sim_utils.DomeLightCfg(
            intensity=lighting_intensity(cfg.ambient_intensity, cfg),
            color=cfg.ambient_color_rgb,
            visible_in_primary_ray=False,
        )
        prims.append(dome_cfg.func(f"{root_path}/UnderwaterAmbient", dome_cfg))
    for light in (cfg.vehicle_lights if include_vehicle_lights else ()):
        light_cfg = sim_utils.DiskLightCfg(
            intensity=lighting_intensity(light.intensity, cfg),
            color=light.color_rgb,
            radius=max(0.005, 0.05 * math.tan(math.radians(light.beam_angle_deg) * 0.5)),
        )
        path = f"{robot_path}/{light.name}"
        prims.append(
            light_cfg.func(
                path,
                light_cfg,
                translation=light.position_b_m,
                orientation=_direction_to_quaternion(light.direction_b),
            )
        )
    return prims


def update_underwater_lighting(
    prims: list[Any],
    cfg: UnderwaterLightingCfg,
    *,
    time_s: float,
    vehicle_distance_m: float = 0.0,
    intensity_scale: float = 1.0,
) -> None:
    """Update spawned USD light intensities without touching physics.

    Isaac Sim versions differ in how light prim handles expose attributes, so
    this helper intentionally uses the USD ``GetAttribute`` protocol and
    quietly ignores a stale handle.  It is safe to call once per render step
    in perception mode and never needs to run in physics-only mode.
    """
    for prim in prims:
        path = str(prim.GetPath()) if hasattr(prim, "GetPath") else ""
        if path.endswith("UnderwaterAmbient"):
            base = cfg.ambient_intensity
            distance = 0.0
        else:
            index = next(
                (i for i, light in enumerate(cfg.vehicle_lights) if path.endswith(f"/{light.name}")),
                None,
            )
            if index is None:
                continue
            base = cfg.vehicle_lights[index].intensity
            distance = vehicle_distance_m
        attribute = _light_intensity_attribute(prim)
        if attribute is not None and attribute.IsValid():
            attribute.Set(
                lighting_intensity(base, cfg, time_s, distance)
                * max(0.0, float(intensity_scale))
            )


def update_underwater_lighting_batch(
    prims_by_env: Sequence[Sequence[Any]],
    cfg: UnderwaterLightingCfg,
    *,
    time_s: float,
    intensity_scales: Any,
    vehicle_distance_m: Any = 0.0,
) -> None:
    """Apply independent per-light intensity scales to cloned environments.

    ``intensity_scales`` is shaped ``[num_envs, num_vehicle_lights]`` and may
    be a CPU/GPU tensor or a nested Python sequence.  USD attribute writes are
    necessarily scalar, but the command generation and clamping remain
    vectorized in the environment.  Invalid or stale prim handles are ignored
    so physics-only tasks can call this helper safely.
    """
    if not prims_by_env or not cfg.enabled or not cfg.vehicle_lights:
        return
    if hasattr(intensity_scales, "detach"):
        scales = intensity_scales.detach().cpu().tolist()
    else:
        scales = intensity_scales
    if hasattr(vehicle_distance_m, "detach"):
        distances = vehicle_distance_m.detach().cpu().reshape(-1).tolist()
    elif isinstance(vehicle_distance_m, Sequence) and not isinstance(vehicle_distance_m, (str, bytes)):
        distances = list(vehicle_distance_m)
    else:
        distances = [float(vehicle_distance_m)] * len(prims_by_env)

    for env_index, prims in enumerate(prims_by_env):
        env_scales = scales[env_index] if env_index < len(scales) else ()
        distance = float(distances[env_index]) if env_index < len(distances) else 0.0
        for prim in prims:
            path = str(prim.GetPath()) if hasattr(prim, "GetPath") else ""
            light_index = next(
                (i for i, light in enumerate(cfg.vehicle_lights) if path.endswith(f"/{light.name}")),
                None,
            )
            if light_index is None:
                continue
            scale = float(env_scales[light_index]) if light_index < len(env_scales) else 1.0
            attribute = _light_intensity_attribute(prim)
            if attribute is not None and attribute.IsValid():
                attribute.Set(
                    lighting_intensity(
                        cfg.vehicle_lights[light_index].intensity,
                        cfg,
                        time_s,
                        distance,
                    )
                    * max(0.0, scale)
                )


def bluerov2_candidate_lighting(*, four_lights: bool = False) -> UnderwaterLightingCfg:
    """Return the source-traceable BlueROV2 active-light candidate.

    Blue Robotics documents two or four 1500-lumen lamps with a 135-degree
    beam.  Exact lamp poses and Isaac photometric calibration are not published
    in the supplied sources, so poses remain explicitly ``estimated``.
    """
    source = "Blue Robotics BlueROV2 datasheet 2025 (standard lighting)"
    lights = [
        VehicleLightCfg(
            name="FrontLightLeft",
            position_b_m=(0.32, 0.12, 0.03),
            direction_b=(1.0, 0.0, 0.0),
            intensity=1500.0,
            beam_angle_deg=135.0,
            source_type="published+estimated_pose",
            source=source,
            confidence="medium_flux_low_pose",
            notes="Flux/beam are manufacturer values; pose is an engineering estimate pending measurement.",
        ),
        VehicleLightCfg(
            name="FrontLightRight",
            position_b_m=(0.32, -0.12, 0.03),
            direction_b=(1.0, 0.0, 0.0),
            intensity=1500.0,
            beam_angle_deg=135.0,
            source_type="published+estimated_pose",
            source=source,
            confidence="medium_flux_low_pose",
            notes="Flux/beam are manufacturer values; pose is an engineering estimate pending measurement.",
        ),
    ]
    if four_lights:
        for side, y in (("Left", 0.12), ("Right", -0.12)):
            lights.append(
                VehicleLightCfg(
                    name=f"RearLight{side}",
                    position_b_m=(-0.30, y, 0.03),
                    direction_b=(1.0, 0.0, 0.0),
                    intensity=1500.0,
                    beam_angle_deg=135.0,
                    source_type="published+estimated_pose",
                    source=source,
                    confidence="medium_flux_low_pose",
                    notes="Four-lamp placement is a provisional symmetric candidate, not measured geometry.",
                )
            )
    return UnderwaterLightingCfg(
        enabled=True,
        ambient_intensity=180.0,
        ambient_color_rgb=(0.08, 0.18, 0.22),
        vehicle_lights=tuple(lights),
    )
