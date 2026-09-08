"""Label-free exploration state used by cave-navigation training.

The policy never receives the visitation grid.  It is privileged training
state used to reward entering a previously unseen workspace voxel and to
report coverage.  Keeping this logic independent from Isaac Sim makes the
reward contract cheap to test and keeps every environment on the same device.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class VoxelVisitTracker:
    """Track per-environment workspace visitation without Python env loops."""

    def __init__(
        self,
        num_envs: int,
        workspace_size_m: Sequence[float],
        voxel_size_m: float,
        device: str | torch.device,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if len(workspace_size_m) != 3 or any(float(size) <= 0.0 for size in workspace_size_m):
            raise ValueError("workspace_size_m must contain three positive values")
        if voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.voxel_size_m = float(voxel_size_m)
        self.workspace_size_m = torch.tensor(workspace_size_m, dtype=torch.float32, device=self.device)
        self.grid_shape = tuple(math.ceil(float(size) / self.voxel_size_m) for size in workspace_size_m)
        self.num_voxels = math.prod(self.grid_shape)
        self.visited = torch.zeros((self.num_envs, self.num_voxels), dtype=torch.bool, device=self.device)
        self.visited_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.new_voxel = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.position_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.path_length_m = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._previous_position_m = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    @property
    def workspace_coverage_fraction(self) -> torch.Tensor:
        """Fraction of the bounding workspace visited, not free-space coverage."""
        return self.visited_count.float() / float(self.num_voxels)

    def _flat_indices(self, relative_position_m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.as_tensor(relative_position_m, dtype=torch.float32, device=self.device)
        if positions.shape != (self.num_envs, 3):
            raise ValueError(f"Expected positions {(self.num_envs, 3)}, got {tuple(positions.shape)}")
        lower = -0.5 * self.workspace_size_m
        coords = torch.floor((positions - lower) / self.voxel_size_m).to(torch.long)
        grid = torch.tensor(self.grid_shape, dtype=torch.long, device=self.device)
        valid = torch.all((coords >= 0) & (coords < grid), dim=-1)
        coords = torch.minimum(torch.maximum(coords, torch.zeros_like(coords)), grid - 1)
        flat = coords[:, 0] * (self.grid_shape[1] * self.grid_shape[2])
        flat += coords[:, 1] * self.grid_shape[2] + coords[:, 2]
        return flat, valid

    def reset(self, env_ids: torch.Tensor | Sequence[int], relative_position_m: torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if relative_position_m.shape != (ids.numel(), 3):
            raise ValueError(f"Expected reset positions {(ids.numel(), 3)}, got {tuple(relative_position_m.shape)}")
        self.visited[ids] = False
        self.visited_count[ids] = 0
        self.new_voxel[ids] = False
        self.position_valid[ids] = False
        self.path_length_m[ids] = 0.0
        self._previous_position_m[ids] = relative_position_m

        # Reuse the batched indexer by placing the reset subset into the full
        # position buffer.  Only ``ids`` are read or mutated below.
        positions = self._previous_position_m
        flat, valid = self._flat_indices(positions)
        valid_ids = ids[valid[ids]]
        if valid_ids.numel() > 0:
            self.visited[valid_ids, flat[valid_ids]] = True
            self.visited_count[valid_ids] = 1
        self.position_valid[ids] = valid[ids]

    def update(self, relative_position_m: torch.Tensor) -> torch.Tensor:
        """Update visitation and return a boolean new-voxel mask."""
        positions = torch.as_tensor(relative_position_m, dtype=torch.float32, device=self.device)
        flat, valid = self._flat_indices(positions)
        env_ids = torch.arange(self.num_envs, device=self.device)
        unseen = valid & ~self.visited[env_ids, flat]
        unseen_ids = env_ids[unseen]
        if unseen_ids.numel() > 0:
            self.visited[unseen_ids, flat[unseen_ids]] = True
            self.visited_count[unseen_ids] += 1
        displacement = torch.linalg.vector_norm(positions - self._previous_position_m, dim=-1)
        self.path_length_m += torch.where(valid, displacement, torch.zeros_like(displacement))
        self._previous_position_m.copy_(positions)
        self.new_voxel.copy_(unseen)
        self.position_valid.copy_(valid)
        return self.new_voxel


def robust_forward_clearance(
    depth: torch.Tensor,
    *,
    max_depth_m: float,
    crop_fraction: float = 0.5,
    quantile: float = 0.1,
) -> torch.Tensor:
    """Return a robust near-obstacle range from the central depth crop.

    Invalid pixels become ``max_depth_m``.  A low quantile is used instead of
    a raw minimum so one bad depth pixel cannot dominate the reward.
    """
    if depth.ndim not in (3, 4):
        raise ValueError(f"Expected depth [N,H,W] or [N,H,W,C], got {tuple(depth.shape)}")
    if max_depth_m <= 0.0:
        raise ValueError("max_depth_m must be positive")
    if not 0.0 < crop_fraction <= 1.0:
        raise ValueError("crop_fraction must be in (0, 1]")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    value = depth[..., 0] if depth.ndim == 4 else depth
    height, width = value.shape[-2:]
    crop_h = max(1, round(height * crop_fraction))
    crop_w = max(1, round(width * crop_fraction))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    value = value[:, y0 : y0 + crop_h, x0 : x0 + crop_w].float()
    valid = torch.isfinite(value) & (value > 0.0)
    value = torch.where(valid, value.clamp(max=max_depth_m), torch.full_like(value, max_depth_m))
    return torch.quantile(value.flatten(start_dim=1), quantile, dim=1)
