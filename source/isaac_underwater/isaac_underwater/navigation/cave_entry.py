"""Geometry-derived cave portal inference for entry curricula.

The active cave preprocessing pipeline produces a provisional skeleton with a
per-sample surface-clearance estimate.  A run of non-finite clearance at a
skeleton endpoint followed by collision-safe finite clearance is an automatic
portal candidate: the endpoint is outside the reconstructed surface envelope
and the transition points into traversable cave geometry.  No hand-authored
entrance coordinate is required.

These helpers are deliberately independent from Isaac Sim.  Portal geometry
is privileged task state and must never be appended to the deployed visual
actor observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CavePortal:
    """One automatically inferred cave opening."""

    endpoint: str
    transition_index: int
    exterior_endpoint_index: int
    chainage_m: float
    exterior_run_m: float
    local_clearance_m: float
    position_m: torch.Tensor
    inward_direction: torch.Tensor


def interpolate_polyline(
    points_m: torch.Tensor,
    chainage_m: torch.Tensor,
    query_chainage_m: torch.Tensor,
) -> torch.Tensor:
    """Linearly interpolate a monotonic polyline for one or more queries."""
    points = torch.as_tensor(points_m)
    if not points.is_floating_point():
        points = points.float()
    chainage = torch.as_tensor(chainage_m, dtype=points.dtype, device=points.device)
    query = torch.as_tensor(query_chainage_m, dtype=points.dtype, device=points.device)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points [N,3], got {tuple(points.shape)}")
    if chainage.ndim != 1 or chainage.shape[0] != points.shape[0]:
        raise ValueError("chainage_m must contain one value per point")
    if points.shape[0] < 2:
        raise ValueError("Polyline needs at least two points")
    if not torch.all(torch.isfinite(points)) or not torch.all(torch.isfinite(chainage)):
        raise ValueError("Polyline points and chainage must be finite")
    if not torch.all(chainage[1:] > chainage[:-1]):
        raise ValueError("chainage_m must be strictly increasing")

    clamped = query.clamp(float(chainage[0]), float(chainage[-1]))
    upper = torch.searchsorted(chainage, clamped, right=False).clamp(1, points.shape[0] - 1)
    lower = upper - 1
    denominator = (chainage[upper] - chainage[lower]).clamp_min(torch.finfo(points.dtype).eps)
    alpha = ((clamped - chainage[lower]) / denominator).unsqueeze(-1)
    return points[lower] + alpha * (points[upper] - points[lower])


def infer_clearance_portals(
    points_m: torch.Tensor,
    chainage_m: torch.Tensor,
    surface_clearance_m: torch.Tensor,
    *,
    minimum_clearance_m: float,
    minimum_exterior_run_m: float = 1.0,
    tangent_probe_m: float = 1.0,
) -> tuple[CavePortal, ...]:
    """Infer endpoint portals from surface-envelope clearance transitions.

    A candidate must start with a contiguous non-finite clearance run.  This
    intentionally rejects a merely narrow/blocked route endpoint: finite but
    small clearance is not evidence that the skeleton lies outside the mesh.
    """
    points = torch.as_tensor(points_m)
    if not points.is_floating_point():
        points = points.float()
    chainage = torch.as_tensor(chainage_m, dtype=points.dtype, device=points.device)
    clearance = torch.as_tensor(surface_clearance_m, dtype=points.dtype, device=points.device)
    # Reuse the interpolation validator and preserve its useful diagnostics.
    interpolate_polyline(points, chainage, chainage[:1])
    if clearance.ndim != 1 or clearance.shape[0] != points.shape[0]:
        raise ValueError("surface_clearance_m must contain one value per point")
    if minimum_clearance_m <= 0.0:
        raise ValueError("minimum_clearance_m must be positive")
    if minimum_exterior_run_m <= 0.0:
        raise ValueError("minimum_exterior_run_m must be positive")
    if tangent_probe_m <= 0.0:
        raise ValueError("tangent_probe_m must be positive")

    portals: list[CavePortal] = []
    count = points.shape[0]
    for endpoint, order, direction in (
        ("start", range(count), 1.0),
        ("end", range(count - 1, -1, -1), -1.0),
    ):
        indices = list(order)
        prefix_length = 0
        while prefix_length < count and not bool(torch.isfinite(clearance[indices[prefix_length]])):
            prefix_length += 1
        # There must be observed exterior and at least one interior sample.
        if prefix_length == 0 or prefix_length == count:
            continue

        transition_offset = prefix_length
        while transition_offset < count:
            value = clearance[indices[transition_offset]]
            if bool(torch.isfinite(value)) and float(value) >= minimum_clearance_m:
                break
            transition_offset += 1
        if transition_offset == count:
            continue

        endpoint_index = indices[0]
        transition_index = indices[transition_offset]
        exterior_run = abs(float(chainage[transition_index] - chainage[endpoint_index]))
        if exterior_run < minimum_exterior_run_m:
            continue

        portal_chainage = float(chainage[transition_index])
        probe_chainage = portal_chainage + direction * tangent_probe_m
        probe_chainage = min(max(probe_chainage, float(chainage[0])), float(chainage[-1]))
        probe = interpolate_polyline(
            points,
            chainage,
            torch.tensor([probe_chainage], dtype=points.dtype, device=points.device),
        )[0]
        inward = probe - points[transition_index]
        norm = torch.linalg.vector_norm(inward)
        if not bool(torch.isfinite(norm)) or float(norm) <= 1.0e-6:
            continue
        portals.append(
            CavePortal(
                endpoint=endpoint,
                transition_index=transition_index,
                exterior_endpoint_index=endpoint_index,
                chainage_m=portal_chainage,
                exterior_run_m=exterior_run,
                local_clearance_m=float(clearance[transition_index]),
                position_m=points[transition_index].detach().clone(),
                inward_direction=(inward / norm).detach().clone(),
            )
        )

    return tuple(portals)
