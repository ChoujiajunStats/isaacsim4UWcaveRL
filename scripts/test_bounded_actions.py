#!/usr/bin/env python3
"""CPU numerical contracts for experimental inference rules."""
import unittest

import torch

from isaac_underwater.learning.bounded_actions import clipped_normal_mean, tanh_normal_mean


class BoundedActionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_variance_symmetry_and_bounds(self):
        mean = torch.linspace(-20, 20, 41, dtype=torch.float64)
        for function, transform in ((clipped_normal_mean, lambda x: x.clamp(-1, 1)),
                                    (tanh_normal_mean, torch.tanh)):
            torch.testing.assert_close(function(mean, torch.zeros_like(mean)), transform(mean))
            result = function(mean, torch.full_like(mean, 2.0))
            torch.testing.assert_close(result, -result.flip(0))
            self.assertTrue(torch.isfinite(result).all() and (result.abs() <= 1).all())

    def test_against_dense_normal_integration(self):
        mean = torch.tensor([-5., -1.2, -.1, 0., .3, 1., 4.], dtype=torch.float64)
        z = torch.linspace(-9, 9, 120001, dtype=torch.float64)
        density = torch.exp(-z.square() / 2) / (2 * torch.pi) ** .5
        for scale in (.001, .2, 1.1, 2., 7.389):
            std = torch.full_like(mean, scale)
            for function, transform, tolerance in (
                (clipped_normal_mean, lambda x: x.clamp(-1, 1), 1.e-7),
                (tanh_normal_mean, torch.tanh, 1.e-7),
            ):
                with self.subTest(scale=scale, function=function.__name__):
                    target = torch.trapezoid(transform(mean[:, None] + std[:, None] * z) * density, z)
                    torch.testing.assert_close(function(mean, std), target, atol=tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
