"""Low-cost constant, sinusoidal, and slowly varying water-current fields."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CurrentMode(str, Enum):
    NONE = "none"
    CONSTANT = "constant"
    RANDOM_CONSTANT = "random_constant"
    SLOW_VARYING = "slow_varying"
    # Backward-compatible names retained for existing configs.
    SINUSOIDAL = "sinusoidal"
    RANDOM_WALK = "random_walk"


@dataclass(frozen=True)
class CurrentProfileCfg:
    mode: CurrentMode | str = CurrentMode.CONSTANT
    mean_velocity_w_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude_w_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    period_s: float = 60.0
    random_walk_std_mps: float = 0.01
    max_speed_mps: float = 0.5


class CurrentField:
    """A vectorized current field with persistent per-environment variation."""

    def __init__(
        self,
        cfg: CurrentProfileCfg,
        num_envs: int,
        device: torch.device | str,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if cfg.period_s <= 0.0:
            raise ValueError("period_s must be positive")
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = num_envs
        self._generator = generator
        self._mean = torch.tensor(cfg.mean_velocity_w_mps, dtype=torch.float32, device=self.device)
        self._amplitude = torch.tensor(cfg.amplitude_w_mps, dtype=torch.float32, device=self.device)
        self._random_velocity = self._mean.expand(num_envs, 3).clone()
        self._phase = torch.empty(num_envs, device=self.device)
        self.reset()

    @property
    def mode(self) -> CurrentMode:
        return CurrentMode(self.cfg.mode)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        if self.mode in (CurrentMode.RANDOM_CONSTANT, CurrentMode.RANDOM_WALK):
            noise = torch.randn(
                (ids.numel(), 3), device=self.device, generator=self._generator
            ) * self.cfg.random_walk_std_mps
            self._random_velocity[ids] = self._mean + noise
        elif self.mode is CurrentMode.NONE:
            self._random_velocity[ids] = 0.0
        else:
            self._random_velocity[ids] = self._mean
        self._phase[ids] = torch.rand(ids.numel(), device=self.device, generator=self._generator) * (2.0 * torch.pi)

    def step(self, dt: float) -> None:
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        if self.mode is not CurrentMode.RANDOM_WALK or dt == 0.0:
            return
        noise = torch.randn(
            self._random_velocity.shape, device=self.device, generator=self._generator
        ) * self.cfg.random_walk_std_mps * (dt**0.5)
        self._random_velocity.add_(noise)
        speed = torch.linalg.vector_norm(self._random_velocity, dim=-1, keepdim=True).clamp_min(1.0e-6)
        self._random_velocity.mul_(torch.clamp(self.cfg.max_speed_mps / speed, max=1.0))

    def velocity(self, time_s: float | torch.Tensor) -> torch.Tensor:
        if self.mode in (CurrentMode.NONE,):
            return torch.zeros(self.num_envs, 3, dtype=self._mean.dtype, device=self.device)
        if self.mode in (CurrentMode.CONSTANT, CurrentMode.RANDOM_CONSTANT, CurrentMode.RANDOM_WALK):
            return self._random_velocity.clone()
        if self.mode in (CurrentMode.SINUSOIDAL, CurrentMode.SLOW_VARYING):
            phase = torch.as_tensor(time_s, device=self.device, dtype=torch.float32)
            oscillation = torch.sin(2.0 * torch.pi * phase / self.cfg.period_s + self._phase)
            return self._mean + oscillation.unsqueeze(-1) * self._amplitude
        return self._random_velocity.clone()
