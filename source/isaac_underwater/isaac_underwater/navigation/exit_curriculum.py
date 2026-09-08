"""Training-only reverse curriculum with an independent frontier per cave."""

from __future__ import annotations

import math
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
        if (
            not all(math.isfinite(value) for value in (initial_distance_m, success_threshold, growth))
            or initial_distance_m <= 0.0 or type(window) is not int or window <= 0
            or not 0.0 < success_threshold <= 1.0 or growth <= 1.0
        ):
            raise ValueError("Invalid exit-curriculum distance, window, threshold, or growth")
        if route_lengths.ndim != 1 or not route_lengths.numel() or not bool(
            torch.all(torch.isfinite(route_lengths) & (route_lengths > 0.0))
        ):
            raise ValueError("Exit curriculum requires a finite positive route length per scene")
        self.route_lengths = route_lengths.clone()
        self.distance_m = self.route_lengths.clamp(max=initial_distance_m)
        self.window = int(window)
        self.success_threshold = float(success_threshold)
        self.growth = float(growth)
        self._outcomes: list[list[bool]] = [[] for _ in route_lengths]

    def state_dict(self) -> dict:
        """Plain checkpoint data; no device-specific tensors or live list aliases."""
        return {
            "version": 1,
            "route_lengths_m": self.route_lengths.tolist(),
            "distance_m": self.distance_m.tolist(),
            "window": self.window,
            "success_threshold": self.success_threshold,
            "growth": self.growth,
            "outcomes": [list(history) for history in self._outcomes],
        }

    def load_state_dict(self, state: dict) -> None:
        """Validate the entire state before changing any active curriculum."""
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported exit-curriculum checkpoint version")
        for key in ("window", "success_threshold", "growth"):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"Exit-curriculum checkpoint/config mismatch: {key}")
        routes = torch.as_tensor(state.get("route_lengths_m"), device=self.route_lengths.device,
                                 dtype=self.route_lengths.dtype)
        distances = torch.as_tensor(state.get("distance_m"), device=self.distance_m.device,
                                    dtype=self.distance_m.dtype)
        if routes.shape != self.route_lengths.shape or not torch.allclose(routes, self.route_lengths):
            raise ValueError("Exit-curriculum checkpoint route lengths do not match this dataset")
        if distances.shape != self.distance_m.shape or not bool(torch.all(
            torch.isfinite(distances) & (distances > 0.0) & (distances <= self.route_lengths + 1.e-4)
        )):
            raise ValueError("Invalid exit-curriculum checkpoint distances")
        outcomes = state.get("outcomes")
        if (
            not isinstance(outcomes, list) or len(outcomes) != len(self._outcomes)
            or any(not isinstance(history, list) or len(history) > self.window
                   or any(type(value) is not bool for value in history) for history in outcomes)
        ):
            raise ValueError("Invalid exit-curriculum checkpoint outcome windows")
        self.distance_m.copy_(distances)
        self._outcomes = [list(history) for history in outcomes]

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
