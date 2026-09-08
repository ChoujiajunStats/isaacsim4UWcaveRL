#!/usr/bin/env python3
"""CPU-only regression checks for the full-route multi-scene acceptance gate."""

from __future__ import annotations

import copy
import unittest

from check_navigation_metrics import validate_metrics


class NavigationMetricsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = {
            "cave_dataset_profile": "train_all",
            "navigation_curriculum": False,
            "domain_randomization": False,
            "episodes": 90,
            "per_scene": {
                scene: {"episodes": 30, "success_rate": 0.9, "collision_rate": 0.1, "mean_spl": 0.8}
                for scene in ("easy", "medium", "hard")
            },
        }

    def validate(self, metrics=None, **kwargs) -> list[str]:
        return validate_metrics(
            self.metrics if metrics is None else metrics,
            expected_profile="train_all", expected_scenes={"easy", "medium", "hard"}, **kwargs,
        )

    def test_complete_full_route_result_passes(self) -> None:
        self.assertEqual(self.validate(), [])

    def test_missing_scene_fails_even_when_profile_and_total_agree(self) -> None:
        del self.metrics["per_scene"]["hard"]
        self.metrics["episodes"] = 60
        self.assertTrue(any("missing scenes" in error for error in self.validate()))

    def test_extra_scene_fails(self) -> None:
        self.metrics["per_scene"]["unknown"] = copy.deepcopy(self.metrics["per_scene"]["easy"])
        self.metrics["episodes"] = 120
        self.assertTrue(any("unexpected scenes" in error for error in self.validate()))

    def test_curriculum_must_be_explicitly_disabled(self) -> None:
        for value in (True, None, "false", 0):
            with self.subTest(value=value):
                self.metrics["navigation_curriculum"] = value
                self.assertTrue(any("navigation_curriculum" in error for error in self.validate()))
        del self.metrics["navigation_curriculum"]
        self.assertTrue(self.validate())

    def test_episode_count_cannot_hide_short_scene_or_mismatch(self) -> None:
        self.metrics["per_scene"]["hard"]["episodes"] = 1
        errors = self.validate()
        self.assertTrue(any("hard: episodes" in error for error in errors))
        self.assertTrue(any("sum of per-scene" in error for error in errors))

    def test_rates_must_be_real_finite_and_within_range(self) -> None:
        for field in ("success_rate", "collision_rate", "mean_spl"):
            for invalid in (True, float("nan"), float("inf"), -0.01, 1.01, "0.9"):
                with self.subTest(field=field, invalid=invalid):
                    metrics = copy.deepcopy(self.metrics)
                    metrics["per_scene"]["easy"][field] = invalid
                    self.assertTrue(self.validate(metrics))

    def test_poor_scene_cannot_be_hidden_by_other_scenes(self) -> None:
        for field, value in (("success_rate", 0.79), ("collision_rate", 0.16), ("mean_spl", 0.49)):
            with self.subTest(field=field):
                metrics = copy.deepcopy(self.metrics)
                metrics["per_scene"]["hard"][field] = value
                self.assertTrue(self.validate(metrics))

    def test_randomization_requires_real_boolean(self) -> None:
        for value in (False, "true", 1, None):
            self.metrics["domain_randomization"] = value
            self.assertTrue(self.validate(require_domain_randomization=True))
        self.metrics["domain_randomization"] = True
        self.assertEqual(self.validate(require_domain_randomization=True), [])

    def test_heldout_profile_accepts_only_its_own_scene(self) -> None:
        self.metrics["cave_dataset_profile"] = "loo_hard_eval"
        self.metrics["per_scene"] = {"hard": self.metrics["per_scene"]["hard"]}
        self.metrics["episodes"] = 30
        self.assertEqual(validate_metrics(
            self.metrics, expected_profile="loo_hard_eval", expected_scenes={"hard"},
        ), [])
        self.assertTrue(self.validate())

    def test_bad_types_and_boolean_counts_fail(self) -> None:
        for metrics in (None, [], {}, {"per_scene": []}, {"per_scene": {}}):
            self.assertTrue(validate_metrics(
                metrics, expected_profile="train_all", expected_scenes={"easy", "medium", "hard"},
            ))
        self.metrics["per_scene"]["easy"]["episodes"] = True
        self.assertTrue(self.validate(min_episodes=1))


if __name__ == "__main__":
    unittest.main()
