#!/usr/bin/env python3
"""CPU contracts using the pinned official FlashSAC, not a substitute SAC."""

from __future__ import annotations

import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch

from isaac_underwater.learning.flash_sac import (
    TerminalObservationMixin, load_flashsac_config, make_flashsac_agent,
    pack_flashsac_observation, replay_storage_bytes,
    restore_flashsac_training_state,
)


def make_agent(*, capacity=64, num_envs=4):
    cfg = load_flashsac_config(
        device="cpu", buffer_max_length=capacity, buffer_min_length=4, sample_batch_size=4,
        actor_hidden_dim=16, critic_hidden_dim=16,
    )
    return make_flashsac_agent(8, 3, num_envs, cfg)


class FakeAutoResetEnvironment:
    device = "cpu"

    def __init__(self):
        self.values = torch.tensor([[1.], [2.]])

    def _get_observations(self):
        return {"policy": self.values.clone(), "critic": 2 * self.values}

    def _reset_idx(self, ids):
        self.values[ids] = -100.

    def step(self, actions):
        self.values += 10
        self._reset_idx(torch.tensor([0]))
        return (self._get_observations(), torch.ones(2), torch.tensor([False, False]),
                torch.tensor([True, False]), {"kept": True})


class CapturingEnvironment(TerminalObservationMixin, FakeAutoResetEnvironment):
    pass


class FlashSACContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_actual_terminal_frame_not_autoreset_frame_and_manual_reset_safe(self):
        env = CapturingEnvironment()
        obs, _, terminated, truncated, extras = env.step(None)
        torch.testing.assert_close(obs["policy"], torch.tensor([[-100.], [12.]]))
        torch.testing.assert_close(extras["final_observation"]["policy"], torch.tensor([[11.], [12.]]))
        torch.testing.assert_close(extras["final_observation"]["critic"], torch.tensor([[22.], [24.]]))
        self.assertTrue(extras["kept"])
        self.assertFalse(terminated.any())
        self.assertEqual(truncated.tolist(), [True, False])
        self.assertFalse(env._capture_terminal_observation)
        env._reset_idx(torch.tensor([1]))
        torch.testing.assert_close(extras["final_observation"]["policy"], torch.tensor([[11.], [12.]]))

    def test_pack_preserves_actor_prefix_and_bounded_replay_budget(self):
        packed = pack_flashsac_observation({"policy": torch.ones(4, 8), "critic": torch.zeros(4, 3)})
        self.assertEqual(packed.shape, (4, 11))
        self.assertTrue(torch.all(packed[:, :8] == 1))
        self.assertTrue(torch.all(packed[:, 8:] == 0))
        self.assertAlmostEqual(replay_storage_bytes(32768, 1572, 6) / 2 ** 20, 394.125)

    def test_rejects_unsupported_configs(self):
        for changes in ({"n_step": 3}, {"use_amp": True}, {"use_compile": True},
                        {"asymmetric_observation": False}, {"sample_batch_size": 1},
                        {"buffer_min_length": 65, "buffer_max_length": 64}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                load_flashsac_config(device="cpu", **changes)

    def test_official_update_asymmetry_terminal_flags_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        source = make_agent()
        observation = torch.randn(4, 11)
        original = [p.detach().clone() for p in source._actor.network.parameters()]
        actions = source.sample_actions(0, {"next_observation": observation}, training=True)
        self.assertEqual(actions.shape, (4, 6))
        self.assertTrue(np.isfinite(actions).all() and (np.abs(actions) <= 1).all())
        next_observation = torch.randn(4, 11) + 7
        source.process_transition({
            "observation": observation, "action": torch.from_numpy(actions),
            "reward": torch.ones(4), "next_observation": next_observation,
            "terminated": torch.tensor([True, False, False, False]),
            "truncated": torch.tensor([False, True, False, False]),
        })
        batch = source._replay_buffer.sample(np.arange(4))
        torch.testing.assert_close(batch["next_observation"], next_observation)
        self.assertEqual(batch["terminated"].tolist(), [1., 0., 0., 0.])
        self.assertEqual(batch["truncated"].tolist(), [0., 1., 0., 0.])
        self.assertTrue(source.can_start_training())
        for _ in range(4):
            metrics = source.update()
            self.assertTrue(all(math.isfinite(value) for value in metrics.values()), metrics)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(original, source._actor.network.parameters())))
        expected = source.sample_actions(0, {"next_observation": observation}, training=False)
        changed_critic = observation.clone()
        changed_critic[:, 8:] += 100
        actual = source.sample_actions(0, {"next_observation": changed_critic}, training=False)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        with TemporaryDirectory(prefix="cave-flashsac-") as directory:
            source.save(directory)
            source.save_replay_buffer(directory)
            target = make_agent()
            target.load(directory)
            target.load_replay_buffer(directory)
        self.assertEqual(target._update_step, 4)
        self.assertEqual(len(target._replay_buffer), 4)
        actual = target.sample_actions(0, {"next_observation": observation}, training=False)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(target._replay_buffer.sample(np.arange(4))["next_observation"], next_observation)
        self.assertTrue(all(math.isfinite(value) for value in target.update().values()))

    def test_resize_resume_preserves_ring_history_and_resets_worker_returns(self):
        for additions in (4, 8, 12):
            with self.subTest(additions=additions):
                source = make_agent(capacity=8)
                for start in range(0, additions, 4):
                    source.process_transition({
                        "observation": torch.ones(4, 11), "next_observation": torch.ones(4, 11) * 2,
                        "action": torch.zeros(4, 6), "reward": torch.arange(start, start + 4, dtype=torch.float32),
                        "terminated": torch.zeros(4, dtype=torch.bool), "truncated": torch.zeros(4, dtype=torch.bool),
                    })
                with TemporaryDirectory(prefix="flashsac-resize-") as directory:
                    source.save(directory)
                    source.save_replay_buffer(directory)
                    target = make_agent(capacity=16, num_envs=6)
                    target.load(directory)
                    restore_flashsac_training_state(target, Path(directory), old_capacity=8, num_envs=6)
                    with self.assertRaises(ValueError):
                        restore_flashsac_training_state(make_agent(capacity=4), Path(directory), old_capacity=8, num_envs=4)
                expected = torch.arange(max(0, additions - 8), additions, dtype=torch.float32)
                torch.testing.assert_close(target._replay_buffer.sample(np.arange(len(expected)))["reward"], expected)
                torch.testing.assert_close(target.reward_normalizer.G_r, torch.zeros(6))
                target.process_transition({
                    "observation": torch.ones(6, 11), "next_observation": torch.ones(6, 11) * 3,
                    "action": torch.zeros(6, 6), "reward": torch.arange(20, 26, dtype=torch.float32),
                    "terminated": torch.zeros(6, dtype=torch.bool), "truncated": torch.zeros(6, dtype=torch.bool),
                })
                all_rewards = target._replay_buffer.sample(np.arange(len(expected) + 6))["reward"]
                torch.testing.assert_close(all_rewards, torch.cat((expected, torch.arange(20, 26, dtype=torch.float32))))
                self.assertTrue(all(math.isfinite(value) for value in target.update().values()))


if __name__ == "__main__":
    unittest.main()
