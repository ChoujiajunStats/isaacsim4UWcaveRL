"""Training-only reverse curriculum with an independent frontier per cave."""

from __future__ import annotations

import torch


class ExitDistanceCurriculum:
    """Increase remaining route distance after a window of successful exits.

    Evaluation does not instantiate this class. Scene frontiers progress
    independently, so a wide cave cannot promote an unlearned narrow cave.
    Completed episodes sampled from earlier frontiers do not count toward a
    newer frontier's promotion window.
    """

    def __init__(
        self,
        route_lengths: torch.Tensor,
        initial_distance_m: float = 4.0,
        window: int = 10,
        success_threshold: float = 0.7,
        growth: float = 1.5,
    ) -> None:
        if initial_distance_m <= 0.0 or window <= 0 or not 0.0 < success_threshold <= 1.0 or growth <= 1.0:
            raise ValueError("Invalid exit-curriculum distance, window, threshold, or growth")
        self.route_lengths = route_lengths.clone()
        self.distance_m = self.route_lengths.clamp(max=initial_distance_m)
        self.window = int(window)
        self.success_threshold = float(success_threshold)
        self.growth = float(growth)
        self._outcomes: list[list[bool]] = [[] for _ in route_lengths]

    def record(self, scene_ids: torch.Tensor, successes: torch.Tensor, episode_frontiers: torch.Tensor) -> None:
        for scene_id, success, frontier in zip(
            scene_ids.tolist(), successes.tolist(), episode_frontiers.tolist()
        ):
            if abs(frontier - float(self.distance_m[scene_id])) > 1.0e-4:
                continue
            history = self._outcomes[scene_id]
            history.append(bool(success))
            if len(history) > self.window:
                del history[0]
            if len(history) == self.window and sum(history) / self.window >= self.success_threshold:
                self.distance_m[scene_id] = torch.minimum(
                    self.distance_m[scene_id] * self.growth, self.route_lengths[scene_id]
                )
                history.clear()
