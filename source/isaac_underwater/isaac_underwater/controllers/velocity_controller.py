"""Future velocity-command abstraction for cave-navigation policies."""

from __future__ import annotations

import torch

from isaac_underwater.interfaces import ControlCommand, ControlMode


class VelocityController:
    def __init__(
        self,
        linear_gain: tuple[float, float, float],
        yaw_gain: float,
        max_force_n: tuple[float, float, float],
        max_yaw_torque_nm: float,
        device: str,
    ):
        self.linear_gain = torch.tensor(linear_gain, dtype=torch.float32, device=device)
        self.yaw_gain = yaw_gain
        self.max_force = torch.tensor(max_force_n, dtype=torch.float32, device=device)
        self.max_yaw_torque = max_yaw_torque_nm

    def compute(
        self,
        desired_velocity_b: torch.Tensor,
        desired_yaw_rate: torch.Tensor,
        velocity_b: torch.Tensor,
        yaw_rate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        force = self.linear_gain * (desired_velocity_b - velocity_b)
        force = torch.maximum(torch.minimum(force, self.max_force), -self.max_force)
        yaw_torque = self.yaw_gain * (desired_yaw_rate - yaw_rate)
        yaw_torque = yaw_torque.clamp(-self.max_yaw_torque, self.max_yaw_torque)
        torque = torch.zeros_like(force)
        torque[:, 2] = yaw_torque
        return force, torque

    def compute_command(
        self,
        command: ControlCommand,
        velocity_b: torch.Tensor,
        angular_velocity_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if command.mode is not ControlMode.BODY_VELOCITY:
            raise ValueError(f"VelocityController cannot process mode {command.mode.value}")
        yaw_rate = angular_velocity_b[:, 2]
        desired_yaw_rate = command.yaw_rate.reshape(-1)
        return self.compute(command.body_velocity, desired_yaw_rate, velocity_b, yaw_rate)
