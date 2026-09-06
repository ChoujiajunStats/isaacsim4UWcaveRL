"""Individual reversible thrusters and six-axis allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .thruster_curve import ThrusterCurve, ThrusterCurveCfg


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class ThrusterCfg:
    name: str
    position_b_m: Vector3
    direction_b: Vector3
    max_forward_thrust_n: float = 40.0
    max_reverse_thrust_n: float = 30.0
    dead_zone: float = 0.05
    curve_exponent: float = 2.0
    response_time_s: float = 0.08
    curve_type: str = "polynomial"
    curve_command_points: tuple[float, ...] = (0.0, 1.0)
    curve_forward_points: tuple[float, ...] = (0.0, 1.0)
    curve_reverse_points: tuple[float, ...] = (0.0, 1.0)


def bluerov2_thruster_layout() -> tuple[ThrusterCfg, ...]:
    """Return a BlueROV2-like eight-thruster layout in x-forward/y-left/z-up."""
    d = 1.0 / math.sqrt(2.0)
    horizontal = (
        ThrusterCfg("front_left", (0.25, 0.18, 0.0), (d, -d, 0.0)),
        ThrusterCfg("front_right", (0.25, -0.18, 0.0), (d, d, 0.0)),
        ThrusterCfg("rear_left", (-0.25, 0.18, 0.0), (d, d, 0.0)),
        ThrusterCfg("rear_right", (-0.25, -0.18, 0.0), (d, -d, 0.0)),
    )
    vertical = (
        ThrusterCfg("vertical_front_left", (0.22, 0.16, 0.0), (0.0, 0.0, 1.0)),
        ThrusterCfg("vertical_front_right", (0.22, -0.16, 0.0), (0.0, 0.0, 1.0)),
        ThrusterCfg("vertical_rear_left", (-0.22, 0.16, 0.0), (0.0, 0.0, 1.0)),
        ThrusterCfg("vertical_rear_right", (-0.22, -0.16, 0.0), (0.0, 0.0, 1.0)),
    )
    return horizontal + vertical


class ThrusterModel:
    """Map normalized commands to individual thrust with lag and saturation."""

    def __init__(self, thrusters: tuple[ThrusterCfg, ...], device: torch.device | str):
        if not thrusters:
            raise ValueError("At least one thruster is required")
        self.cfg = thrusters
        self.device = torch.device(device)
        self.positions = torch.tensor([item.position_b_m for item in thrusters], dtype=torch.float32, device=device)
        directions = torch.tensor([item.direction_b for item in thrusters], dtype=torch.float32, device=device)
        self.directions = torch.nn.functional.normalize(directions, dim=-1)
        self.forward = torch.tensor([item.max_forward_thrust_n for item in thrusters], device=device)
        self.reverse = torch.tensor([item.max_reverse_thrust_n for item in thrusters], device=device)
        self.dead_zone = torch.tensor([item.dead_zone for item in thrusters], device=device)
        self.exponent = torch.tensor([item.curve_exponent for item in thrusters], device=device)
        curve_types = {item.curve_type.lower() for item in thrusters}
        if len(curve_types) != 1:
            raise ValueError("A vectorized ThrusterModel requires one curve type for all thrusters")
        self.curve = ThrusterCurve(
            ThrusterCurveCfg(
                kind=next(iter(curve_types)),
                exponent=thrusters[0].curve_exponent,
                command_points=thrusters[0].curve_command_points,
                forward_points=thrusters[0].curve_forward_points,
                reverse_points=thrusters[0].curve_reverse_points,
            ),
            device,
        )
        self.time_constant = torch.tensor([item.response_time_s for item in thrusters], device=device)
        self.thrust_state: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return len(self.cfg)

    def reset(self, batch_size: int, env_ids: torch.Tensor | None = None) -> None:
        if self.thrust_state is None or self.thrust_state.shape[0] != batch_size:
            self.thrust_state = torch.zeros(batch_size, self.count, device=self.device)
        elif env_ids is None:
            self.thrust_state.zero_()
        else:
            self.thrust_state[env_ids] = 0.0

    def command_to_target_thrust(
        self,
        command: torch.Tensor,
        *,
        dead_zone: torch.Tensor | None = None,
    ) -> torch.Tensor:
        command = command.clamp(-1.0, 1.0)
        dead_zone_t = self.dead_zone if dead_zone is None else torch.as_tensor(
            dead_zone, dtype=command.dtype, device=command.device
        ).reshape(-1, 1)
        magnitude = command.abs()
        active = magnitude > dead_zone_t
        scaled = ((magnitude - dead_zone_t) / (1.0 - dead_zone_t)).clamp_min(0.0)
        scaled = self.curve.evaluate(scaled, reverse=command < 0.0)
        limit = torch.where(command >= 0.0, self.forward, self.reverse)
        return torch.where(active, command.sign() * scaled * limit, torch.zeros_like(command))

    def step(
        self,
        command: torch.Tensor,
        dt: float,
        *,
        dead_zone: torch.Tensor | None = None,
        response_time_s: torch.Tensor | None = None,
        failure_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if command.shape[-1] != self.count:
            raise ValueError(f"Expected {self.count} thruster commands, got {command.shape[-1]}")
        if self.thrust_state is None or self.thrust_state.shape != command.shape:
            self.reset(command.shape[0])
        target = self.command_to_target_thrust(command, dead_zone=dead_zone)
        if failure_mask is not None:
            target = torch.where(
                torch.as_tensor(failure_mask, dtype=torch.bool, device=command.device),
                torch.zeros_like(target),
                target,
            )
        if response_time_s is None:
            time_constant = self.time_constant
        else:
            time_constant = torch.as_tensor(
                response_time_s, dtype=command.dtype, device=command.device
            ).reshape(-1, 1)
        alpha = torch.where(
            time_constant > 0.0,
            1.0 - torch.exp(-torch.as_tensor(dt, device=self.device) / time_constant),
            torch.ones_like(time_constant),
        )
        self.thrust_state = self.thrust_state + alpha * (target - self.thrust_state)
        return self.thrust_state

    def wrench_from_thrust(self, thrust_n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        individual_forces = thrust_n.unsqueeze(-1) * self.directions
        individual_torques = torch.linalg.cross(self.positions.expand_as(individual_forces), individual_forces, dim=-1)
        return individual_forces.sum(dim=-2), individual_torques.sum(dim=-2)


class ThrusterAllocator:
    """Least-squares allocation from body wrench to reversible commands."""

    def __init__(self, model: ThrusterModel, rcond: float = 1.0e-5):
        self.model = model
        moments = torch.linalg.cross(model.positions, model.directions, dim=-1)
        self.allocation_matrix = torch.cat((model.directions.T, moments.T), dim=0)
        if torch.linalg.matrix_rank(self.allocation_matrix).item() < 6:
            raise ValueError("Thruster layout does not span all six body wrench axes")
        self.pseudoinverse = torch.linalg.pinv(self.allocation_matrix, rcond=rcond)

    def allocate_thrust(self, force_b: torch.Tensor, torque_b: torch.Tensor) -> torch.Tensor:
        wrench = torch.cat((force_b, torque_b), dim=-1)
        return wrench @ self.pseudoinverse.T

    def allocate_command(self, force_b: torch.Tensor, torque_b: torch.Tensor) -> torch.Tensor:
        thrust = self.allocate_thrust(force_b, torque_b)
        limit = torch.where(thrust >= 0.0, self.model.forward, self.model.reverse)
        normalized = (thrust.abs() / limit).clamp(0.0, 1.0)
        normalized = self.model.curve.inverse(normalized, reverse=thrust < 0.0)
        command_magnitude = self.model.dead_zone + (1.0 - self.model.dead_zone) * normalized
        return torch.where(thrust == 0.0, torch.zeros_like(thrust), thrust.sign() * command_magnitude).clamp(-1.0, 1.0)
