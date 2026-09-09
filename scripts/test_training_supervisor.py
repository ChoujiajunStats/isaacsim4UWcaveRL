#!/usr/bin/env python3
"""CPU tests for resource guard semantics; no simulator or GPU writes."""

import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from supervise_training import ResourceGuard, memory_fields, run_guarded
from run_dual_navigation_training import transition_budget


class TrainingSupervisorTest(unittest.TestCase):
    def test_child_completion_and_owned_group_resource_stop(self):
        for available, expected in ((2000, "complete"), (500, "failed")):
            with self.subTest(available=available), TemporaryDirectory(prefix="training-supervisor-") as directory:
                sample = {"host_available_mb": available, "process_rss_mb": 20, "swap_used_mb": 0,
                          "gpu_memory_mb": 100, "gpu_temperature_c": 50, "gpu_util_percent": 0}
                code = "print('child completed')" if expected == "complete" else "import time; time.sleep(5)"
                with patch("supervise_training.sample_resources", return_value=sample):
                    result = run_guarded([sys.executable, "-c", code], Path(directory) / "child.log",
                                         max_seconds=10, guard=ResourceGuard(required_samples=1), poll_seconds=.01)
                self.assertEqual(result["state"], expected)
                self.assertEqual(result["stop_reason"], None if expected == "complete" else "low_host_available_memory")
                self.assertIsInstance(result["return_code"], int)

    def test_equal_transition_budgets_allow_different_worker_counts(self):
        for ppo_envs, flash_envs in ((48, 72), (96, 72), (72, 96)):
            budget = transition_budget(2000000, ppo_envs, flash_envs)
            self.assertGreaterEqual(budget, 2000000)
            self.assertEqual(budget % (ppo_envs * 64), 0)
            self.assertEqual(budget % flash_envs, 0)
        with self.assertRaises(ValueError):
            transition_budget(2000000, 50, 72)

    def test_meminfo_units_and_partial_process_status(self):
        self.assertEqual(memory_fields("MemAvailable: 2048000 kB\nSwapTotal: 1024000 kB\nSwapFree: 512000 kB\n"),
                         {"MemAvailable": 2000., "SwapTotal": 1000., "SwapFree": 500.})
        self.assertEqual(memory_fields("Name: python\nVmRSS: 1024 kB\n"), {"VmRSS": 1.})

    def test_transient_pressure_does_not_stop_but_persistent_pressure_does(self):
        guard = ResourceGuard()
        low = {"host_available_mb": 500, "gpu_memory_mb": 4000, "gpu_temperature_c": 70}
        safe = {**low, "host_available_mb": 2000}
        self.assertIsNone(guard.check(low))
        self.assertIsNone(guard.check(low))
        self.assertIsNone(guard.check(safe))
        self.assertIsNone(guard.check(low))
        self.assertIsNone(guard.check(low))
        self.assertEqual(guard.check(low), "low_host_available_memory")

    def test_temperature_and_gpu_headroom_are_independent_limits(self):
        guard = ResourceGuard(required_samples=1)
        self.assertEqual(guard.check({"host_available_mb": 2000, "gpu_memory_mb": 7800, "gpu_temperature_c": 85}),
                         "gpu_memory_headroom,gpu_temperature")


if __name__ == "__main__":
    unittest.main()
