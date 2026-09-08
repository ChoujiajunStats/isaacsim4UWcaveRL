#!/usr/bin/env python3
"""CPU-only checks that parallel evaluation cannot over-sample fast failures."""

import unittest

from isaac_underwater.navigation.evaluation_quota import SceneEpisodeQuota


class EvaluationQuotaTest(unittest.TestCase):
    def test_balances_unequal_worker_counts(self):
        scenes = [0, 1, 2, 0, 1, 2, 0, 1]
        quota = SceneEpisodeQuota(scenes, 3, 10)
        self.assertEqual(quota.quotas, [4, 4, 5, 3, 3, 5, 3, 3])
        for scene in range(3):
            values = [quota.quotas[i] for i, value in enumerate(scenes) if value == scene]
            self.assertEqual(sum(values), 10)
            self.assertLessEqual(max(values) - min(values), 1)

    def test_fast_failures_cannot_steal_slow_worker_quota(self):
        quota = SceneEpisodeQuota([0, 0, 1, 1], 2, 4)
        self.assertEqual(sum(quota.accept(0, 0) for _ in range(100)), 2)
        self.assertEqual(sum(quota.accept(2, 1) for _ in range(100)), 2)
        self.assertEqual(sum(quota.counts), 4)
        self.assertEqual(sum(quota.accept(1, 0) for _ in range(2)), 2)
        self.assertEqual(sum(quota.accept(3, 1) for _ in range(2)), 2)
        self.assertEqual(sum(quota.counts), 8)

    def test_more_workers_than_requested_episodes(self):
        quota = SceneEpisodeQuota([0, 1, 0, 1, 0, 1], 2, 1)
        self.assertEqual(quota.quotas, [1, 1, 0, 0, 0, 0])
        self.assertFalse(quota.accept(4, 0))

    def test_single_worker_per_scene(self):
        self.assertEqual(SceneEpisodeQuota([0, 1, 2], 3, 30).quotas, [30, 30, 30])

    def test_bad_coverage_and_terminal_identity_fail(self):
        for scenes, count, episodes in (([], 1, 1), ([0], 2, 1), ([0, 2], 2, 1),
                                        ([True], 1, 1), ([0], 1, 0), ([0], 0, 1)):
            with self.subTest(scenes=scenes, count=count, episodes=episodes):
                with self.assertRaises(ValueError):
                    SceneEpisodeQuota(scenes, count, episodes)
        quota = SceneEpisodeQuota([0, 1], 2, 3)
        for env, scene in ((0, 1), (-1, 0), (2, 0)):
            with self.assertRaises(ValueError):
                quota.accept(env, scene)


if __name__ == "__main__":
    unittest.main()
