"""Explicit sim-to-real gap categories and vectorized sampling helpers."""

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


Range = tuple[float, float]


@dataclass(frozen=True)
class DynamicsGapCfg:
    mass_scale: Range = (1.0, 1.0)
    inertia_scale: Range = (1.0, 1.0)
    added_mass_scale: Range = (1.0, 1.0)
    buoyancy_scale: Range = (1.0, 1.0)
    center_offset_m: Range = (0.0, 0.0)
    drag_scale: Range = (1.0, 1.0)
    current_speed_mps: Range = (0.0, 0.0)


@dataclass(frozen=True)
class ActuatorGapCfg:
    strength_scale: Range = (1.0, 1.0)
    asymmetry_scale: Range = (1.0, 1.0)
    dead_zone: Range = (0.05, 0.05)
    latency_s: Range = (0.0, 0.0)
    response_time_s: Range = (0.08, 0.08)
    command_noise_std: Range = (0.0, 0.0)
    saturation_scale: Range = (1.0, 1.0)


@dataclass(frozen=True)
class SensorGapCfg:
    imu_bias_std: Range = (0.0, 0.0)
    imu_noise_std: Range = (0.0, 0.0)
    camera_noise_std: Range = (0.0, 0.0)
    exposure_offset: Range = (0.0, 0.0)
    timestamp_jitter_s: Range = (0.0, 0.0)
    frame_drop_probability: Range = (0.0, 0.0)
    camera_imu_offset_s: Range = (0.0, 0.0)
    extrinsic_rotation_deg: Range = (0.0, 0.0)
    depth_noise_std: Range = (0.0, 0.0)
    motion_blur_strength: Range = (0.0, 0.0)


@dataclass(frozen=True)
class AppearanceGapCfg:
    visibility_range_m: Range = (12.0, 12.0)
    ambient_scale: Range = (1.0, 1.0)
    attenuation_scale: Range = (1.0, 1.0)
    backscatter_strength: Range = (0.0, 0.0)
    texture_scale: Range = (1.0, 1.0)


@dataclass(frozen=True)
class LocalizationGapCfg:
    position_drift_mps: Range = (0.0, 0.0)
    orientation_drift_radps: Range = (0.0, 0.0)
    dropout_probability: Range = (0.0, 0.0)
    relocalization_delay_s: Range = (0.0, 0.0)


@dataclass(frozen=True)
class GeometryGapCfg:
    scale: Range = (1.0, 1.0)
    obstacle_density: Range = (0.0, 0.0)


@dataclass(frozen=True)
class DomainGapCfg:
    dynamics: DynamicsGapCfg = field(default_factory=DynamicsGapCfg)
    actuator: ActuatorGapCfg = field(default_factory=ActuatorGapCfg)
    sensor: SensorGapCfg = field(default_factory=SensorGapCfg)
    appearance: AppearanceGapCfg = field(default_factory=AppearanceGapCfg)
    localization: LocalizationGapCfg = field(default_factory=LocalizationGapCfg)
    geometry: GeometryGapCfg = field(default_factory=GeometryGapCfg)


def _sample_range(bounds: Range, count: int, device: torch.device | str, generator=None) -> torch.Tensor:
    low, high = bounds
    if high < low:
        raise ValueError(f"Invalid randomization range: {bounds}")
    return torch.empty(count, device=device).uniform_(low, high, generator=generator)


def sample_domain_gap(
    cfg: DomainGapCfg,
    count: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample one independent parameter vector per environment.

    The returned names are stable experiment keys, making sampled conditions
    straightforward to log alongside an episode and replay in sim-to-real
    studies.  The PointNav task applies the actuator, hydrodynamics, and sensor
    subsets explicitly while preserving this generic sampler for other tasks.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    samples = {
        "mass_scale": _sample_range(cfg.dynamics.mass_scale, count, device, generator),
        "inertia_scale": _sample_range(cfg.dynamics.inertia_scale, count, device, generator),
        "added_mass_scale": _sample_range(cfg.dynamics.added_mass_scale, count, device, generator),
        "buoyancy_scale": _sample_range(cfg.dynamics.buoyancy_scale, count, device, generator),
        "center_offset_m": _sample_range(cfg.dynamics.center_offset_m, count, device, generator),
        "drag_scale": _sample_range(cfg.dynamics.drag_scale, count, device, generator),
        "current_speed_mps": _sample_range(cfg.dynamics.current_speed_mps, count, device, generator),
        "thruster_strength_scale": _sample_range(cfg.actuator.strength_scale, count, device, generator),
        "thruster_asymmetry_scale": _sample_range(cfg.actuator.asymmetry_scale, count, device, generator),
        "dead_zone": _sample_range(cfg.actuator.dead_zone, count, device, generator),
        "response_time_s": _sample_range(cfg.actuator.response_time_s, count, device, generator),
        "actuator_latency_s": _sample_range(cfg.actuator.latency_s, count, device, generator),
        "command_noise_std": _sample_range(cfg.actuator.command_noise_std, count, device, generator),
        "actuator_saturation_scale": _sample_range(cfg.actuator.saturation_scale, count, device, generator),
        "imu_bias_std": _sample_range(cfg.sensor.imu_bias_std, count, device, generator),
        "imu_noise_std": _sample_range(cfg.sensor.imu_noise_std, count, device, generator),
        "camera_noise_std": _sample_range(cfg.sensor.camera_noise_std, count, device, generator),
        "depth_noise_std": _sample_range(cfg.sensor.depth_noise_std, count, device, generator),
        "motion_blur_strength": _sample_range(cfg.sensor.motion_blur_strength, count, device, generator),
        "exposure_offset": _sample_range(cfg.sensor.exposure_offset, count, device, generator),
        "timestamp_jitter_s": _sample_range(cfg.sensor.timestamp_jitter_s, count, device, generator),
        "frame_drop_probability": _sample_range(cfg.sensor.frame_drop_probability, count, device, generator),
        "camera_imu_offset_s": _sample_range(cfg.sensor.camera_imu_offset_s, count, device, generator),
        "extrinsic_rotation_deg": _sample_range(cfg.sensor.extrinsic_rotation_deg, count, device, generator),
        "visibility_range_m": _sample_range(cfg.appearance.visibility_range_m, count, device, generator),
        "ambient_scale": _sample_range(cfg.appearance.ambient_scale, count, device, generator),
        "attenuation_scale": _sample_range(cfg.appearance.attenuation_scale, count, device, generator),
        "backscatter_strength": _sample_range(cfg.appearance.backscatter_strength, count, device, generator),
        "texture_scale": _sample_range(cfg.appearance.texture_scale, count, device, generator),
        "position_drift_mps": _sample_range(cfg.localization.position_drift_mps, count, device, generator),
        "orientation_drift_radps": _sample_range(cfg.localization.orientation_drift_radps, count, device, generator),
        "dropout_probability": _sample_range(cfg.localization.dropout_probability, count, device, generator),
        "relocalization_delay_s": _sample_range(cfg.localization.relocalization_delay_s, count, device, generator),
        "geometry_scale": _sample_range(cfg.geometry.scale, count, device, generator),
        "obstacle_density": _sample_range(cfg.geometry.obstacle_density, count, device, generator),
    }
    validity = validate_physical_samples(samples)
    if not bool(validity.all()):
        invalid = (~validity).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Domain-gap sampler produced physically invalid environments: {invalid}")
    return samples


def validate_physical_samples(samples: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return one validity flag per environment before applying a sample.

    Scales that multiply mass, inertia, volume, damping, or actuator lag must
    stay positive.  The checker is deliberately tensorized so reset-time
    resampling can remain vectorized when richer randomization is added.
    """
    if not samples:
        raise ValueError("samples must not be empty")
    first = next(iter(samples.values()))
    valid = torch.ones(first.shape[0], dtype=torch.bool, device=first.device)
    valid &= torch.isfinite(first)
    for key, value in samples.items():
        valid &= torch.isfinite(value).all(dim=tuple(range(1, value.ndim))) if value.ndim > 1 else torch.isfinite(value)
        if key in {
            "mass_scale", "inertia_scale", "added_mass_scale", "buoyancy_scale",
            "drag_scale", "thruster_strength_scale", "response_time_s", "actuator_saturation_scale",
        }:
            valid &= value > 0.0
        if key in {"dead_zone", "frame_drop_probability", "dropout_probability"}:
            valid &= (value >= 0.0) & (value < 1.0)
    return valid


def _range(value: Any, name: str) -> Range:
    """Normalize a YAML two-element range into the typed config contract."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element range, got {value!r}")
    result = (float(value[0]), float(value[1]))
    if result[1] < result[0]:
        raise ValueError(f"{name} has descending bounds: {result}")
    return result


def domain_gap_from_mapping(mapping: Mapping[str, Any]) -> DomainGapCfg:
    """Build a :class:`DomainGapCfg` from ``configs/rl/domain_gap.yaml``.

    Keeping this conversion here means an experiment can use the exact same
    YAML contract from a task, a standalone sampler, or a real-robot replay
    tool without importing Isaac Lab.
    """
    dynamics = mapping.get("dynamics", {})
    actuator = mapping.get("actuator", {})
    sensor = mapping.get("sensor", {})
    appearance = mapping.get("appearance", {})
    localization = mapping.get("localization", {})
    geometry = mapping.get("geometry", {})
    return DomainGapCfg(
        dynamics=DynamicsGapCfg(
            mass_scale=_range(dynamics.get("mass_scale", (1.0, 1.0)), "dynamics.mass_scale"),
            inertia_scale=_range(dynamics.get("inertia_scale", (1.0, 1.0)), "dynamics.inertia_scale"),
            added_mass_scale=_range(dynamics.get("added_mass_scale", (1.0, 1.0)), "dynamics.added_mass_scale"),
            buoyancy_scale=_range(dynamics.get("buoyancy_scale", (1.0, 1.0)), "dynamics.buoyancy_scale"),
            center_offset_m=_range(dynamics.get("center_offset_m", (0.0, 0.0)), "dynamics.center_offset_m"),
            drag_scale=_range(dynamics.get("drag_scale", (1.0, 1.0)), "dynamics.drag_scale"),
            current_speed_mps=_range(
                dynamics.get("current_speed_mps", (0.0, 0.0)), "dynamics.current_speed_mps"
            ),
        ),
        actuator=ActuatorGapCfg(
            strength_scale=_range(actuator.get("strength_scale", (1.0, 1.0)), "actuator.strength_scale"),
            asymmetry_scale=_range(actuator.get("asymmetry_scale", (1.0, 1.0)), "actuator.asymmetry_scale"),
            dead_zone=_range(actuator.get("dead_zone", (0.05, 0.05)), "actuator.dead_zone"),
            latency_s=_range(actuator.get("latency_s", (0.0, 0.0)), "actuator.latency_s"),
            response_time_s=_range(
                actuator.get("response_time_s", (0.08, 0.08)), "actuator.response_time_s"
            ),
            command_noise_std=_range(
                actuator.get("command_noise_std", (0.0, 0.0)), "actuator.command_noise_std"
            ),
            saturation_scale=_range(
                actuator.get("saturation_scale", (1.0, 1.0)), "actuator.saturation_scale"
            ),
        ),
        sensor=SensorGapCfg(
            imu_bias_std=_range(sensor.get("imu_bias_std", (0.0, 0.0)), "sensor.imu_bias_std"),
            imu_noise_std=_range(sensor.get("imu_noise_std", (0.0, 0.0)), "sensor.imu_noise_std"),
            camera_noise_std=_range(
                sensor.get("camera_noise_std", (0.0, 0.0)), "sensor.camera_noise_std"
            ),
            exposure_offset=_range(sensor.get("exposure_offset", (0.0, 0.0)), "sensor.exposure_offset"),
            timestamp_jitter_s=_range(
                sensor.get("timestamp_jitter_s", (0.0, 0.0)), "sensor.timestamp_jitter_s"
            ),
            frame_drop_probability=_range(
                sensor.get("frame_drop_probability", (0.0, 0.0)), "sensor.frame_drop_probability"
            ),
            camera_imu_offset_s=_range(
                sensor.get("camera_imu_offset_s", (0.0, 0.0)), "sensor.camera_imu_offset_s"
            ),
            extrinsic_rotation_deg=_range(
                sensor.get("extrinsic_rotation_deg", (0.0, 0.0)), "sensor.extrinsic_rotation_deg"
            ),
            depth_noise_std=_range(sensor.get("depth_noise_std", (0.0, 0.0)), "sensor.depth_noise_std"),
            motion_blur_strength=_range(
                sensor.get("motion_blur_strength", (0.0, 0.0)), "sensor.motion_blur_strength"
            ),
        ),
        appearance=AppearanceGapCfg(
            visibility_range_m=_range(
                appearance.get("visibility_range_m", (12.0, 12.0)), "appearance.visibility_range_m"
            ),
            ambient_scale=_range(appearance.get("ambient_scale", (1.0, 1.0)), "appearance.ambient_scale"),
            attenuation_scale=_range(
                appearance.get("attenuation_scale", (1.0, 1.0)), "appearance.attenuation_scale"
            ),
            backscatter_strength=_range(
                appearance.get("backscatter_strength", (0.0, 0.0)), "appearance.backscatter_strength"
            ),
            texture_scale=_range(appearance.get("texture_scale", (1.0, 1.0)), "appearance.texture_scale"),
        ),
        localization=LocalizationGapCfg(
            position_drift_mps=_range(
                localization.get("position_drift_mps", (0.0, 0.0)), "localization.position_drift_mps"
            ),
            orientation_drift_radps=_range(
                localization.get("orientation_drift_radps", (0.0, 0.0)),
                "localization.orientation_drift_radps",
            ),
            dropout_probability=_range(
                localization.get("dropout_probability", (0.0, 0.0)), "localization.dropout_probability"
            ),
            relocalization_delay_s=_range(
                localization.get("relocalization_delay_s", (0.0, 0.0)),
                "localization.relocalization_delay_s",
            ),
        ),
        geometry=GeometryGapCfg(
            scale=_range(geometry.get("scale", (1.0, 1.0)), "geometry.scale"),
            obstacle_density=_range(
                geometry.get("obstacle_density", (0.0, 0.0)), "geometry.obstacle_density"
            ),
        ),
    )


def perturb_thruster_command(
    command: torch.Tensor,
    samples: dict[str, torch.Tensor],
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply sampled actuator strength, noise, asymmetry, and saturation.

    ``samples`` is the output of :func:`sample_domain_gap`, so the operation is
    deterministic when a caller supplies a seeded generator and the sampled
    state is logged with an episode.
    """
    batch = command.shape[0]
    expand = lambda key: samples[key].reshape(batch, 1)
    perturbed = command * expand("thruster_strength_scale")
    asymmetry = expand("thruster_asymmetry_scale")
    perturbed = torch.where(perturbed >= 0.0, perturbed * asymmetry, perturbed / asymmetry.clamp_min(1.0e-6))
    noise = torch.randn(perturbed.shape, device=command.device, generator=generator)
    perturbed = perturbed + noise * expand("command_noise_std")
    return perturbed.clamp(-expand("actuator_saturation_scale"), expand("actuator_saturation_scale"))


def rotate_body_vectors_z(vectors: torch.Tensor, angle_deg: torch.Tensor) -> torch.Tensor:
    """Apply a per-environment yaw extrinsic perturbation to 3-vectors."""
    if vectors.shape[-1] != 3:
        raise ValueError(f"vectors must end in three coordinates, got {vectors.shape}")
    angle = torch.as_tensor(angle_deg, device=vectors.device, dtype=vectors.dtype)
    if vectors.ndim == 1 and angle.ndim == 0:
        angle = torch.deg2rad(angle)
    else:
        angle = torch.deg2rad(angle).reshape(-1, *([1] * (vectors.ndim - 2)))
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    x, y, z = vectors.unbind(dim=-1)
    return torch.stack((cosine * x - sine * y, sine * x + cosine * y, z), dim=-1)
