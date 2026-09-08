"""Fixed episode allocation for balanced, finite parallel cave evaluation."""

from __future__ import annotations

from collections import Counter


class SceneEpisodeQuota:
    """Fast-ending workers cannot consume slower workers' evaluation samples."""

    def __init__(self, scene_ids: list[int], num_scenes: int, episodes_per_scene: int):
        if type(num_scenes) is not int or num_scenes <= 0:
            raise ValueError("num_scenes must be a positive integer")
        if type(episodes_per_scene) is not int or episodes_per_scene <= 0:
            raise ValueError("episodes_per_scene must be a positive integer")
        if any(type(scene_id) is not int for scene_id in scene_ids) or set(scene_ids) != set(range(num_scenes)):
            raise ValueError("Evaluation workers must cover exactly the selected scenes")
        self.scene_ids = list(scene_ids)
        workers_per_scene = Counter(scene_ids)
        ranks = Counter()
        self.quotas: list[int] = []
        for scene_id in scene_ids:
            base, remainder = divmod(episodes_per_scene, workers_per_scene[scene_id])
            self.quotas.append(base + int(ranks[scene_id] < remainder))
            ranks[scene_id] += 1
        self.counts = [0] * len(scene_ids)

    def accept(self, env_id: int, scene_id: int) -> bool:
        if not 0 <= env_id < len(self.scene_ids) or self.scene_ids[env_id] != scene_id:
            raise ValueError("Terminal episode scene does not match its evaluation worker")
        if self.counts[env_id] >= self.quotas[env_id]:
            return False
        self.counts[env_id] += 1
        return True
