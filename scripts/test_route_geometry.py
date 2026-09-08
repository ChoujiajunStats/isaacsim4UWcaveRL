#!/usr/bin/env python3
"""Regression checks for continuous navigation progress and reverse curriculum."""

import torch

from isaac_underwater.navigation.exit_curriculum import ExitDistanceCurriculum
from isaac_underwater.navigation.route_geometry import project_polyline, sample_polyline


def main() -> None:
    points = torch.tensor([
        [[0., 0., 0.], [10., 0., 0.], [10., 5., 0.], [0., 0., 0.]],
        [[20., 0., 1.], [20., 3., 5.], [0., 0., 0.], [0., 0., 0.]],
    ])
    chainages = torch.tensor([[0., 10., 15., 0.], [0., 5., 0., 0.]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    positions = torch.tensor([[0.05, 0.2, 0.], [20., 1.5, 3.]])
    distance, progress, _ = project_polyline(positions, points, chainages, mask)
    torch.testing.assert_close(distance, torch.tensor([0.2, 0.]))
    torch.testing.assert_close(progress, torch.tensor([0.05, 2.5]))
    # The first 5 cm of a 10 m segment must receive progress; a nearest-vertex
    # approximation would report zero and use the wrong corridor distance.
    positions[0, 0] += 0.05
    _, next_progress, _ = project_polyline(positions, points, chainages, mask)
    torch.testing.assert_close(next_progress - progress, torch.tensor([0.05, 0.]))
    positions[0] = torch.tensor([9.8, 2., 0.])
    distance, progress, _ = project_polyline(positions, points, chainages, mask)
    torch.testing.assert_close(distance, torch.tensor([0.2, 0.]))
    torch.testing.assert_close(progress, torch.tensor([12., 2.5]))
    target, tangent = sample_polyline(torch.tensor([12., 5.]), points, chainages, mask)
    torch.testing.assert_close(target, torch.tensor([[10., 2., 0.], [20., 3., 5.]]))
    torch.testing.assert_close(tangent, torch.tensor([[0., 1., 0.], [0., 0.6, 0.8]]))
    # Moving the complete parallel environment cannot change local metrics.
    offset = torch.tensor([200., -200., 7.])
    shifted = project_polyline(positions + offset, points + offset, chainages, mask)
    torch.testing.assert_close(shifted[0], distance, atol=2.e-5, rtol=1.e-5)
    torch.testing.assert_close(shifted[1], progress)

    curriculum = ExitDistanceCurriculum(torch.tensor([9., 20.]), window=3, success_threshold=2/3)
    for success in (True, False, True):
        curriculum.record(torch.tensor([0, 1]), torch.tensor([success, False]), torch.tensor([4., 4.]))
    torch.testing.assert_close(curriculum.distance_m, torch.tensor([6., 4.]))
    # Successes from an old, easier frontier cannot promote the new frontier.
    for _ in range(3):
        curriculum.record(torch.tensor([0]), torch.tensor([True]), torch.tensor([4.]))
    torch.testing.assert_close(curriculum.distance_m, torch.tensor([6., 4.]))
    for _ in range(3):
        curriculum.record(torch.tensor([0]), torch.tensor([True]), torch.tensor([6.]))
    torch.testing.assert_close(curriculum.distance_m, torch.tensor([9., 4.]))
    for _ in range(3):
        curriculum.record(torch.tensor([0]), torch.tensor([True]), torch.tensor([9.]))
    torch.testing.assert_close(curriculum.distance_m, torch.tensor([9., 4.]))
    print("continuous_route_and_curriculum: PASS")


if __name__ == "__main__":
    main()
