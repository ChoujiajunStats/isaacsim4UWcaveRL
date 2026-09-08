"""A lightweight stereo CNN + GRU actor with a privileged recurrent critic.

RSL-RL 3.1 resolves policy names in ``on_policy_runner``'s module namespace.
``register_rsl_rl_extensions`` adds this project-owned class to that namespace
without editing the installed RSL-RL package.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCriticRecurrent
from rsl_rl.networks import EmpiricalNormalization, Memory
from tensordict import TensorDict


class StereoFeatureEncoder(nn.Module):
    """Encode flattened NCHW stereo RGB/depth plus scalar measurements."""

    def __init__(
        self,
        raw_dim: int,
        image_shape: Sequence[int],
        visual_latent_dim: int,
        scalar_latent_dim: int,
        normalize_scalars: bool,
    ) -> None:
        super().__init__()
        if len(image_shape) != 3:
            raise ValueError("image_shape must be (channels, height, width)")
        self.image_shape = tuple(int(value) for value in image_shape)
        if any(value <= 0 for value in self.image_shape):
            raise ValueError("image_shape values must be positive")
        self.image_dim = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        self.scalar_dim = int(raw_dim) - self.image_dim
        if self.scalar_dim <= 0:
            raise ValueError(
                f"Visual observation {raw_dim} must contain {self.image_dim} image values and scalar inputs"
            )
        if visual_latent_dim <= 0 or scalar_latent_dim <= 0:
            raise ValueError("latent dimensions must be positive")

        self.scalar_normalizer = (
            EmpiricalNormalization(self.scalar_dim) if normalize_scalars else nn.Identity()
        )
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(self.image_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(32 * 3 * 4, visual_latent_dim),
            nn.ELU(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(self.scalar_dim, scalar_latent_dim),
            nn.ELU(),
        )
        self.output_dim = int(visual_latent_dim + scalar_latent_dim)

    def _split(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
        if observation.shape[-1] != self.image_dim + self.scalar_dim:
            raise ValueError(
                f"Expected raw observation dimension {self.image_dim + self.scalar_dim}, "
                f"got {observation.shape[-1]}"
            )
        leading_shape = observation.shape[:-1]
        image = observation[..., : self.image_dim].reshape(-1, *self.image_shape)
        scalars = observation[..., self.image_dim :]
        return image, scalars, leading_shape

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        image, scalars, leading_shape = self._split(observation)
        visual_features = self.visual_encoder(image).reshape(*leading_shape, -1)
        scalar_features = self.scalar_encoder(self.scalar_normalizer(scalars))
        return torch.cat((visual_features, scalar_features), dim=-1)

    @torch.jit.unused
    def update(self, observation: torch.Tensor) -> None:
        if isinstance(self.scalar_normalizer, EmpiricalNormalization):
            _, scalars, _ = self._split(observation)
            self.scalar_normalizer.update(scalars.reshape(-1, self.scalar_dim))


class StereoVisualActorCriticRecurrent(ActorCriticRecurrent):
    """RSL-RL-compatible CNN/GRU policy for flattened stereo observations."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = True,
        critic_obs_normalization: bool = True,
        actor_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        critic_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "elu",
        init_noise_std: float = 0.8,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 128,
        rnn_num_layers: int = 1,
        visual_channels: int = 8,
        visual_height: int = 12,
        visual_width: int = 16,
        visual_latent_dim: int = 64,
        scalar_latent_dim: int = 32,
        **kwargs: object,
    ) -> None:
        # Let the upstream class establish the complete PPO distribution,
        # critic, normalization, and checkpoint contract.  Its temporary raw
        # actor memory is immediately replaced by the encoded-memory path.
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=False,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            state_dependent_std=state_dependent_std,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            **kwargs,
        )
        raw_actor_dim = sum(int(obs[group].shape[-1]) for group in obs_groups["policy"])
        self.raw_actor_obs_dim = raw_actor_dim
        self.actor_obs_normalization = True
        self.actor_obs_normalizer = StereoFeatureEncoder(
            raw_actor_dim,
            (visual_channels, visual_height, visual_width),
            visual_latent_dim,
            scalar_latent_dim,
            normalize_scalars=actor_obs_normalization,
        )
        self.memory_a = Memory(
            self.actor_obs_normalizer.output_dim,
            rnn_hidden_dim,
            rnn_num_layers,
            rnn_type,
        )
        print(f"Actor stereo encoder: {self.actor_obs_normalizer}")
        print(f"Actor encoded RNN: {self.memory_a}")


def register_rsl_rl_extensions() -> None:
    """Expose project policies to RSL-RL's name-based policy resolver."""
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    on_policy_runner.StereoVisualActorCriticRecurrent = StereoVisualActorCriticRecurrent
