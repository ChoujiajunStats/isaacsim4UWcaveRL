"""Replaceable normalized thruster command curves."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ThrusterCurveCfg:
    kind: str = "polynomial"
    exponent: float = 2.0
    command_points: tuple[float, ...] = (0.0, 1.0)
    forward_points: tuple[float, ...] = (0.0, 1.0)
    reverse_points: tuple[float, ...] = (0.0, 1.0)


class ThrusterCurve:
    """Evaluate linear, polynomial, lookup, or measured normalized curves."""

    def __init__(self, cfg: ThrusterCurveCfg, device: torch.device | str):
        kind = cfg.kind.lower()
        if kind == "measured_curve":
            kind = "lookup_table"
        if kind not in {"linear", "polynomial", "lookup_table"}:
            raise ValueError(f"Unknown thruster curve type: {cfg.kind}")
        points = torch.tensor(cfg.command_points, dtype=torch.float32, device=device)
        forward = torch.tensor(cfg.forward_points, dtype=torch.float32, device=device)
        reverse = torch.tensor(cfg.reverse_points, dtype=torch.float32, device=device)
        if points.ndim != 1 or forward.shape != points.shape or reverse.shape != points.shape:
            raise ValueError("Thruster lookup points must have equal one-dimensional shapes")
        if points.numel() < 2 or not torch.all(points[1:] > points[:-1]):
            raise ValueError("Thruster lookup command points must be strictly increasing")
        self.kind = kind
        self.exponent = torch.as_tensor(cfg.exponent, dtype=torch.float32, device=device)
        self.command_points = points
        self.forward_points = forward
        self.reverse_points = reverse

    def _lookup(self, magnitude: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        x = self.command_points
        index = torch.bucketize(magnitude, x).clamp(1, x.numel() - 1)
        x0, x1 = x[index - 1], x[index]
        y0, y1 = points[index - 1], points[index]
        fraction = ((magnitude - x0) / (x1 - x0)).clamp(0.0, 1.0)
        return y0 + fraction * (y1 - y0)

    def evaluate(self, magnitude: torch.Tensor, *, reverse: torch.Tensor | None = None) -> torch.Tensor:
        magnitude = magnitude.clamp(0.0, 1.0)
        if self.kind == "linear":
            return magnitude
        if self.kind == "polynomial":
            return magnitude.pow(self.exponent)
        forward = self._lookup(magnitude, self.forward_points)
        if reverse is None:
            return forward
        backward = self._lookup(magnitude, self.reverse_points)
        return torch.where(reverse, backward, forward)

    def inverse(self, thrust_fraction: torch.Tensor, *, reverse: torch.Tensor | None = None) -> torch.Tensor:
        thrust_fraction = thrust_fraction.clamp(0.0, 1.0)
        if self.kind == "linear":
            return thrust_fraction
        if self.kind == "polynomial":
            return thrust_fraction.pow(1.0 / self.exponent.clamp_min(1.0e-6))
        forward_distances = (thrust_fraction.unsqueeze(-1) - self.forward_points).abs()
        forward = self.command_points[forward_distances.argmin(dim=-1)]
        if reverse is None:
            return forward
        reverse_distances = (thrust_fraction.unsqueeze(-1) - self.reverse_points).abs()
        backward = self.command_points[reverse_distances.argmin(dim=-1)]
        return torch.where(reverse, backward, forward)
