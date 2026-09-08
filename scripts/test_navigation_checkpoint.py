#!/usr/bin/env python3
"""CPU checks of curriculum state and actual upstream PPO checkpoint I/O."""

from __future__ import annotations

import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from isaac_underwater.learning.navigation_runner import NavigationOnPolicyRunner
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from isaac_underwater.navigation.exit_curriculum import ExitDistanceCurriculum


class FakePolicy(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.resets = 0

    def reset(self):
        self.resets += 1


class FakeEnvironment:
    def __init__(self, *, enabled=True, scene_keys=("easy", "hard")):
        self.unwrapped = self
        self._exit_curriculum = ExitDistanceCurriculum(torch.tensor([10., 20.]), window=3) if enabled else None
        self._cave_scene_variants = [SimpleNamespace(key=key) for key in scene_keys]
        self.resets = 0
        self.spawn_frontiers = None

    def reset(self):
        self.resets += 1
        self.spawn_frontiers = self._exit_curriculum.distance_m.clone()


def make_runner(**kwargs):
    runner = NavigationOnPolicyRunner.__new__(NavigationOnPolicyRunner)
    runner.env = FakeEnvironment(**kwargs)
    policy = FakePolicy()
    runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()))
    runner.current_learning_iteration = 17
    runner.logger_type = "tensorboard"
    runner.disable_logs = False
    return runner


class NavigationCheckpointTest(unittest.TestCase):
    def test_curriculum_starts_real_episodes_without_artificial_timeout_failures(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), patch.object(OnPolicyRunner, "learn") as learn:
                make_runner(enabled=enabled).learn(3, init_at_random_ep_len=True)
                learn.assert_called_once_with(3, init_at_random_ep_len=not enabled)

    def test_roundtrip_preserves_frontier_pending_window_and_upstream_metadata(self):
        source = make_runner()
        curriculum = source.env._exit_curriculum
        for _ in range(3):
            curriculum.record(torch.tensor([0, 1]), torch.tensor([True, False]), torch.tensor([4., 4.]))
        for _ in range(2):
            curriculum.record(torch.tensor([0]), torch.tensor([True]), torch.tensor([6.]))
        with TemporaryDirectory(prefix="navigation-checkpoint-") as directory:
            path = str(Path(directory) / "model.pt")
            source.save(path, {"caller_metadata": "kept"})
            target = make_runner()
            infos = target.load(path, map_location="cpu")
        self.assertEqual(infos["caller_metadata"], "kept")
        self.assertEqual(target.current_learning_iteration, 17)
        for key, value in source.alg.policy.state_dict().items():
            torch.testing.assert_close(target.alg.policy.state_dict()[key], value)
        self.assertEqual(target.env._exit_curriculum.state_dict(), curriculum.state_dict())
        torch.testing.assert_close(target.env.spawn_frontiers, torch.tensor([6., 4.]))
        self.assertEqual(target.env.resets, 1)
        self.assertEqual(target.alg.policy.resets, 1)
        target.env._exit_curriculum.record(torch.tensor([0]), torch.tensor([True]), torch.tensor([6.]))
        torch.testing.assert_close(target.env._exit_curriculum.distance_m, torch.tensor([9., 4.]))

    def test_old_checkpoint_warns_and_keeps_initial_curriculum(self):
        with TemporaryDirectory(prefix="navigation-checkpoint-") as directory:
            path = str(Path(directory) / "old.pt")
            make_runner(enabled=False).save(path)
            target = make_runner()
            with self.assertWarnsRegex(UserWarning, "no navigation curriculum state"):
                target.load(path, map_location="cpu")
        torch.testing.assert_close(target.env._exit_curriculum.distance_m, torch.tensor([4., 4.]))
        self.assertEqual(target.env.resets, 0)

    def test_evaluation_and_weights_only_load_do_not_restore_curriculum(self):
        with TemporaryDirectory(prefix="navigation-checkpoint-") as directory:
            path = str(Path(directory) / "model.pt")
            source = make_runner()
            source.env._exit_curriculum.distance_m.fill_(6.)
            source.save(path)
            evaluation = make_runner(enabled=False)
            evaluation.load(path, map_location="cpu")
            self.assertIsNone(evaluation.env._exit_curriculum)
            self.assertEqual(evaluation.env.resets, 0)
            warm_start = make_runner()
            warm_start.load(path, load_optimizer=False, map_location="cpu")
            torch.testing.assert_close(warm_start.env._exit_curriculum.distance_m, torch.tensor([4., 4.]))

    def test_profile_order_mismatch_fails(self):
        with TemporaryDirectory(prefix="navigation-checkpoint-") as directory:
            path = str(Path(directory) / "model.pt")
            make_runner().save(path)
            target = make_runner(scene_keys=("hard", "easy"))
            with self.assertRaisesRegex(ValueError, "scene identities/order"):
                target.load(path, map_location="cpu")
        self.assertEqual(target.env.resets, 0)

    def test_invalid_state_is_rejected_without_partial_mutation(self):
        curriculum = ExitDistanceCurriculum(torch.tensor([10., 20.]), window=3)
        original = curriculum.state_dict()
        for key, value in (
            ("version", 99), ("window", 4), ("growth", 2.),
            ("route_lengths_m", [11., 20.]), ("distance_m", [12., 4.]),
            ("distance_m", [float("nan"), 4.]), ("distance_m", [4.]),
            ("outcomes", [[True], [1]]), ("outcomes", [[True] * 4, []]),
        ):
            with self.subTest(key=key, value=value):
                state = copy.deepcopy(original)
                state[key] = value
                with self.assertRaises(ValueError):
                    curriculum.load_state_dict(state)
                self.assertEqual(curriculum.state_dict(), original)

    def test_state_does_not_alias_mutable_histories(self):
        curriculum = ExitDistanceCurriculum(torch.tensor([10., 20.]), window=3)
        state = curriculum.state_dict()
        state["outcomes"][0].append(True)
        self.assertEqual(curriculum.state_dict()["outcomes"], [[], []])
        curriculum.load_state_dict(state)
        state["outcomes"][0].append(False)
        self.assertEqual(curriculum.state_dict()["outcomes"], [[True], []])


if __name__ == "__main__":
    unittest.main()
