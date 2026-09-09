"""PPO configuration for the stereo visual actor / privileged critic pilot."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class StereoVisualActorCriticRecurrentCfg(RslRlPpoActorCriticRecurrentCfg):
    """Configuration fields consumed by the project stereo CNN/GRU policy."""

    class_name: str = "StereoVisualActorCriticRecurrent"
    visual_channels: int = 8
    visual_height: int = 12
    visual_width: int = 16
    visual_latent_dim: int = 64
    scalar_latent_dim: int = 32


@configclass
class UnderwaterVisualPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 500
    save_interval = 25
    experiment_name = "underwater_visual_cave"
    seed = 42
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnderwaterExplorePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Recurrent visual PPO for the label-free cave exploration curriculum."""

    num_steps_per_env = 16
    max_iterations = 500
    save_interval = 25
    experiment_name = "underwater_cave_explore"
    seed = 42
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    policy = StereoVisualActorCriticRecurrentCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[128, 64],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        visual_channels=8,
        visual_height=12,
        visual_width=16,
        visual_latent_dim=64,
        scalar_latent_dim=32,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        # Recurrent mini-batches partition environments, not flattened
        # timesteps.  Keep this compatible with the 2-env RTX 4060 gate.
        num_mini_batches=2,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnderwaterCaveEntryPPORunnerCfg(UnderwaterExplorePPORunnerCfg):
    """Separate log namespace for the geometry-inferred entry curriculum."""

    experiment_name = "underwater_cave_entry"


@configclass
class UnderwaterMultiCaveNavigationPPORunnerCfg(UnderwaterExplorePPORunnerCfg):
    """Long-horizon recurrent PPO shared by all selected cave scenes."""

    num_steps_per_env = 64
    max_iterations = 2000
    save_interval = 50
    experiment_name = "underwater_cave_multinav"
    # Explicit fine-tuning controls. Defaults preserve normal checkpoint resume.
    navigation_weights_only: bool = False
    # Isaac Lab from_dict validates against the default's runtime type; a
    # float sentinel permits Hydra overrides, unlike a None default.
    navigation_reset_noise_std: float = 0.0
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        # train_all defaults to six envs (two per cave), so three recurrent
        # mini-batches preserve scene balance inside each update.
        num_mini_batches=3,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
