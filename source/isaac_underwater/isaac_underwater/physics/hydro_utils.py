"""Small tensor utilities shared by the marine dynamics components."""

from __future__ import annotations

import torch


def skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the batched cross-product matrix ``S(v)``."""
    if vector.shape[-1] != 3:
        raise ValueError(f"Expected a 3-vector, got {tuple(vector.shape)}")
    x, y, z = vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def _normalized_quaternion(quat_wxyz: torch.Tensor) -> torch.Tensor:
    return quat_wxyz / torch.linalg.vector_norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(1.0e-8)


def rotate_world_to_body(quat_wxyz: torch.Tensor, vectors_w: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame vectors by the inverse of a wxyz quaternion."""
    quat_wxyz = _normalized_quaternion(quat_wxyz)
    xyz = quat_wxyz[..., 1:]
    cross = 2.0 * torch.linalg.cross(xyz, vectors_w, dim=-1)
    return vectors_w - quat_wxyz[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def rotate_body_to_world(quat_wxyz: torch.Tensor, vectors_b: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame vectors by a wxyz quaternion."""
    quat_wxyz = _normalized_quaternion(quat_wxyz)
    xyz = quat_wxyz[..., 1:]
    cross = 2.0 * torch.linalg.cross(xyz, vectors_b, dim=-1)
    return vectors_b + quat_wxyz[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def as_batch_scale(
    value: torch.Tensor | float | None,
    batch_shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    default: float = 1.0,
) -> torch.Tensor:
    """Normalize a scalar or per-environment scale without Python loops."""
    if value is None:
        return torch.full(batch_shape, default, dtype=dtype, device=device)
    scale = torch.as_tensor(value, dtype=dtype, device=device)
    if scale.numel() == 1:
        return scale.reshape(()).expand(batch_shape)
    if tuple(scale.shape) != tuple(batch_shape):
        raise ValueError(f"Expected scale shape {batch_shape}, got {tuple(scale.shape)}")
    return scale


def validate_symmetric_positive_definite(matrix: torch.Tensor, *, tolerance: float = 1.0e-6) -> None:
    """Raise ``ValueError`` when a mass matrix violates physical constraints."""
    if matrix.shape[-2:] != (6, 6):
        raise ValueError(f"Expected a 6x6 matrix, got {tuple(matrix.shape)}")
    if not torch.isfinite(matrix).all():
        raise ValueError("Mass matrix contains NaN or Inf")
    if not torch.allclose(matrix, matrix.transpose(-1, -2), atol=tolerance, rtol=tolerance):
        raise ValueError("Mass matrix must be symmetric")
    eigenvalues = torch.linalg.eigvalsh(matrix)
    if torch.any(eigenvalues <= tolerance):
        raise ValueError(f"Mass matrix is not positive definite: {eigenvalues}")
