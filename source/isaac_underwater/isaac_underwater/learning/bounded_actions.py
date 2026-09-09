"""Diagnostic deterministic expectations of bounded Gaussian actions.

These are alternative inference rules, NOT the default PPO/FlashSAC policy
and not a substitute for training or entrance-to-exit acceptance evaluation.
"""

from functools import lru_cache
import math

import numpy as np
import torch


def clipped_normal_mean(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Exact E[clip(N(mean, std**2), -1, 1)] for finite, nonnegative std."""
    safe_std = std.clamp_min(1.e-8)
    lower, upper = (-1.0 - mean) / safe_std, (1.0 - mean) / safe_std
    cdf_lower = 0.5 * (1.0 + torch.erf(lower / math.sqrt(2.0)))
    cdf_upper = 0.5 * (1.0 + torch.erf(upper / math.sqrt(2.0)))
    density_delta = (torch.exp(-0.5 * lower.square()) - torch.exp(-0.5 * upper.square())) / math.sqrt(2 * math.pi)
    result = -cdf_lower + (1 - cdf_upper) + mean * (cdf_upper - cdf_lower) + safe_std * density_delta
    return torch.where(std <= 1.e-8, mean.clamp(-1, 1), result).clamp(-1, 1)


@lru_cache(maxsize=8)
def _normal_quadrature(device: torch.device, dtype: torch.dtype):
    nodes, weights = np.polynomial.legendre.leggauss(512)
    nodes = nodes * 9.0  # Gaussian tail mass outside [-9, 9] is below 3e-19.
    weights = weights * 9.0 * np.exp(-0.5 * nodes ** 2) / math.sqrt(2 * math.pi)
    return (torch.as_tensor(nodes, device=device, dtype=dtype),
            torch.as_tensor(weights, device=device, dtype=dtype))


def tanh_normal_mean(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """512-point quadrature for E[tanh(N(mean, std**2))]; tested to std=exp(2)."""
    nodes, weights = _normal_quadrature(mean.device, mean.dtype)
    return (torch.tanh(mean.unsqueeze(-1) + std.unsqueeze(-1) * nodes) * weights).sum(-1).clamp(-1, 1)
