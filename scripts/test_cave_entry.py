#!/usr/bin/env python3
"""Unit checks for geometry-derived cave portal inference."""

from __future__ import annotations

import torch

from isaac_underwater.navigation import infer_clearance_portals, interpolate_polyline


def main() -> None:
    chainage = torch.arange(0.0, 7.0)
    points = torch.stack((chainage, torch.zeros_like(chainage), 0.1 * chainage), dim=-1)
    clearance = torch.tensor((float("nan"), float("nan"), float("nan"), 0.9, 1.0, 0.8, 0.2))

    portals = infer_clearance_portals(
        points,
        chainage,
        clearance,
        minimum_clearance_m=0.6,
        minimum_exterior_run_m=1.0,
        tangent_probe_m=1.5,
    )
    assert len(portals) == 1
    portal = portals[0]
    assert portal.endpoint == "start"
    assert portal.transition_index == 3
    assert portal.exterior_run_m == 3.0
    torch.testing.assert_close(
        portal.inward_direction,
        torch.tensor((1.0, 0.0, 0.1)) / torch.linalg.vector_norm(torch.tensor((1.0, 0.0, 0.1))),
    )

    queries = torch.tensor((0.5, 2.25, 8.0))
    interpolated = interpolate_polyline(points, chainage, queries)
    torch.testing.assert_close(interpolated[:, 0], torch.tensor((0.5, 2.25, 6.0)))
    torch.testing.assert_close(interpolated[:, 2], torch.tensor((0.05, 0.225, 0.6)))

    two_sided = torch.tensor((float("nan"), float("nan"), 0.9, 1.0, 0.9, float("nan"), float("nan")))
    two_portals = infer_clearance_portals(
        points,
        chainage,
        two_sided,
        minimum_clearance_m=0.6,
        minimum_exterior_run_m=1.0,
    )
    assert [portal.endpoint for portal in two_portals] == ["start", "end"]
    assert two_portals[0].inward_direction[0] > 0.0
    assert two_portals[1].inward_direction[0] < 0.0

    # A narrow but finite endpoint is not automatically classified as open.
    no_exterior = torch.tensor((0.1, 0.2, 0.9, 1.0, 1.0, 0.8, 0.2))
    assert not infer_clearance_portals(
        points,
        chainage,
        no_exterior,
        minimum_clearance_m=0.6,
    )
    print("cave_entry_contract: PASS")


if __name__ == "__main__":
    main()
