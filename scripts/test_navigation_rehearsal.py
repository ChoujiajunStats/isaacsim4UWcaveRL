#!/usr/bin/env python3
"""Stateless rehearsal sampling and frontier-promotion contracts."""
import unittest

import torch

from isaac_underwater.navigation.exit_curriculum import (
    ExitDistanceCurriculum, rehearsal_settings, sample_rehearsal_distances,
)


class RehearsalTest(unittest.TestCase):
    def test_disabled_preserves_values_and_rng(self):
        frontier = torch.tensor([1., 8., 30.])
        rng = torch.random.get_rng_state().clone()
        result = sample_rehearsal_distances(frontier, **rehearsal_settings(None))
        self.assertIs(result, frontier)
        torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)

    def test_bounds_probability_and_reproducibility(self):
        torch.manual_seed(42)
        frontier = torch.full((10000,), 40.)
        settings = dict(probability=.5, min_distance_m=2., max_distance_m=12.)
        result = sample_rehearsal_distances(frontier, **settings)
        self.assertTrue(.48 < (result == frontier).float().mean() < .52)
        shorter = result[result < frontier]
        self.assertTrue(((shorter >= 2) & (shorter <= 12)).all())
        torch.manual_seed(42)
        torch.testing.assert_close(sample_rehearsal_distances(frontier, **settings), result, rtol=0, atol=0)
        tiny = sample_rehearsal_distances(torch.tensor([1., 3.]), **settings)
        self.assertTrue(((tiny > 0) & (tiny <= torch.tensor([1., 3.]))).all())

    def test_rehearsal_cannot_promote_frontier(self):
        curriculum = ExitDistanceCurriculum(torch.tensor([40.]), initial_distance_m=20., window=3)
        curriculum.record(torch.zeros(100, dtype=torch.long), torch.ones(100, dtype=torch.bool), torch.full((100,), 4.))
        self.assertEqual(curriculum.distance_m.item(), 20.)
        self.assertEqual(curriculum.state_dict()["outcomes"], [[]])
        curriculum.record(torch.zeros(3, dtype=torch.long), torch.ones(3, dtype=torch.bool), torch.full((3,), 20.))
        self.assertEqual(curriculum.distance_m.item(), 30.)

    def test_invalid_settings(self):
        for settings in (dict(probability=2., min_distance_m=2., max_distance_m=12.),
                         dict(probability=.5, min_distance_m=0., max_distance_m=12.),
                         dict(probability=.5, min_distance_m=2., max_distance_m=1.),
                         dict(probability=float("nan"), min_distance_m=2., max_distance_m=12.)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                sample_rehearsal_distances(torch.tensor([4.]), **settings)


if __name__ == "__main__":
    unittest.main()
