"""Batched, continuous measurements on padded cave reference polylines."""

from __future__ import annotations

import torch


def project_polyline(
    positions: torch.Tensor,
    points: torch.Tensor,
    chainages: torch.Tensor,
    valid_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return distance, chainage, and segment index for each position.

    Inputs have shapes (N, 3), (N, P, 3), (N, P), and (N, P). Routes must
    contain at least two consecutive valid, distinct points with increasing
    chainage; the route loader validates these conditions before simulation.
    Padding must be marked invalid. The closest point lies on a segment, not
    necessarily on one of the exported vertices, so even centimetre-scale
    motion receives a progress signal on a sparsely sampled straight route.
    """
    starts = points[:, :-1]
    segments = points[:, 1:] - starts
    segment_length_sq = segments.square().sum(dim=-1)
    fraction = (
        ((positions[:, None] - starts) * segments).sum(dim=-1)
        / segment_length_sq.clamp_min(1.0e-12)
    ).clamp(0.0, 1.0)
    projected = starts + fraction[..., None] * segments
    valid_segments = valid_points[:, :-1] & valid_points[:, 1:] & (segment_length_sq > 1.0e-12)
    distance_sq = (positions[:, None] - projected).square().sum(dim=-1)
    distance_sq = distance_sq.masked_fill(~valid_segments, float("inf"))
    nearest_distance_sq, index = distance_sq.min(dim=1)
    projected_chainage = chainages[:, :-1] + fraction * (chainages[:, 1:] - chainages[:, :-1])
    chainage = projected_chainage.gather(1, index[:, None]).squeeze(1)
    return nearest_distance_sq.clamp_min(0.0).sqrt(), chainage, index


def sample_polyline(
    query: torch.Tensor,
    points: torch.Tensor,
    chainages: torch.Tensor,
    valid_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return interpolated position and unit tangent at batched chainages.

    Query values have shape (N,) and must fall inside each route's valid
    range. The remaining inputs have the same shapes as project_polyline().
    """
    valid_segments = valid_points[:, :-1] & valid_points[:, 1:]
    index = ((chainages[:, 1:] < query[:, None]) & valid_segments).sum(dim=1)
    last_segment = valid_segments.sum(dim=1) - 1
    index = torch.minimum(index, last_segment)
    rows = torch.arange(points.shape[0], device=points.device)
    start = points[rows, index]
    segment = points[rows, index + 1] - start
    length = chainages[rows, index + 1] - chainages[rows, index]
    fraction = ((query - chainages[rows, index]) / length.clamp_min(1.0e-12)).clamp(0.0, 1.0)
    tangent = segment / torch.linalg.vector_norm(segment, dim=-1, keepdim=True).clamp_min(1.0e-12)
    return start + fraction[:, None] * segment, tangent
