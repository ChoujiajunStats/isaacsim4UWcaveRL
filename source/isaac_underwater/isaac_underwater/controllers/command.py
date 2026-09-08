"""Shared conversion from high-level control contracts to thruster commands."""

from __future__ import annotations

import torch

from isaac_underwater.actuators import ThrusterAllocator
from isaac_underwater.interfaces import ControlCommand, ControlMode

from .velocity_controller import VelocityController


def command_to_thruster(
    command: ControlCommand,
    *,
    velocity_controller: VelocityController,
    allocator: ThrusterAllocator,
    linear_velocity_b: torch.Tensor,
    yaw_rate: torch.Tensor,
) -> torch.Tensor:
    """Resolve velocity, wrench, or direct-thruster commands uniformly."""
    if command.mode is ControlMode.THRUSTER:
        assert command.thruster_command is not None
        return command.thruster_command.clamp(-1.0, 1.0)
    if command.mode is ControlMode.BODY_WRENCH:
        assert command.body_wrench is not None
        return allocator.allocate_command(command.body_wrench[..., :3], command.body_wrench[..., 3:])
    assert command.body_velocity is not None and command.yaw_rate is not None
    force, torque = velocity_controller.compute(
        command.body_velocity,
        command.yaw_rate,
        linear_velocity_b,
        yaw_rate,
    )
    return allocator.allocate_command(force, torque)
