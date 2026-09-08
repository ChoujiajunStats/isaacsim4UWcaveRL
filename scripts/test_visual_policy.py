#!/usr/bin/env python3
"""CPU contract check for the project stereo CNN + GRU RSL-RL policy."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from isaac_underwater.learning import StereoVisualActorCriticRecurrent, register_rsl_rl_extensions


def main() -> None:
    num_envs = 2
    observations = TensorDict(
        {
            "policy": torch.rand(num_envs, 1549),
            "critic": torch.rand(num_envs, 21),
        },
        batch_size=[num_envs],
    )
    policy = StereoVisualActorCriticRecurrent(
        observations,
        {"policy": ["policy"], "critic": ["critic"]},
        6,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        rnn_type="gru",
        rnn_hidden_dim=32,
        rnn_num_layers=1,
    )
    policy.train()
    policy.update_normalization(observations)
    actions = policy.act(observations)
    values = policy.evaluate(observations)
    assert actions.shape == (num_envs, 6)
    assert values.shape == (num_envs, 1)
    assert torch.isfinite(actions).all() and torch.isfinite(values).all()
    hidden_actor, hidden_critic = policy.get_hidden_states()
    assert hidden_actor is not None and hidden_critic is not None
    policy.reset(torch.tensor([True, False]))

    # Exercise gradients through both the CNN and GRU outside the rollout's
    # inference context.
    policy.reset()
    encoded = policy.actor_obs_normalizer(observations["policy"])
    recurrent, _ = policy.memory_a.rnn(encoded.unsqueeze(0))
    mean_action = policy.actor(recurrent.squeeze(0))
    loss = mean_action.square().mean()
    loss.backward()
    conv_grad = policy.actor_obs_normalizer.visual_encoder[0].weight.grad
    assert conv_grad is not None and torch.isfinite(conv_grad).all()

    register_rsl_rl_extensions()
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    assert on_policy_runner.StereoVisualActorCriticRecurrent is StereoVisualActorCriticRecurrent
    print("visual_policy_contract: PASS")


if __name__ == "__main__":
    main()
