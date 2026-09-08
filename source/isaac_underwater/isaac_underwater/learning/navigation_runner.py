"""Keep reverse-curriculum state in the same checkpoint as upstream PPO."""

from __future__ import annotations

import warnings

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class NavigationOnPolicyRunner(OnPolicyRunner):
    """Only extend checkpoint metadata; optimization remains upstream RSL-RL."""

    def _navigation_env(self):
        env = getattr(self.env, "unwrapped", self.env)
        return env if getattr(env, "_exit_curriculum", None) is not None else None

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # The upstream 300 s random clock can exceed a short curriculum's
        # horizon immediately. Those one-step artificial timeouts would be
        # recorded as failures in the promotion window after every resume.
        if self._navigation_env() is not None:
            init_at_random_ep_len = False
        super().learn(num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)

    def save(self, path: str, infos: dict | None = None) -> None:
        env = self._navigation_env()
        if env is not None:
            infos = dict(infos or {})
            infos["navigation_training"] = {
                "scene_keys": [scene.key for scene in env._cave_scene_variants],
                "exit_curriculum": env._exit_curriculum.state_dict(),
            }
        super().save(path, infos)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        env = self._navigation_env()
        # Evaluation has no curriculum. A weights-only warm start deliberately
        # keeps the caller's freshly configured curriculum instead of restoring.
        if env is None or not load_optimizer:
            return infos
        state = (infos or {}).get("navigation_training")
        if state is None:
            warnings.warn(
                "Checkpoint has no navigation curriculum state; using configured initial distances.",
                stacklevel=2,
            )
            return infos
        scene_keys = [scene.key for scene in env._cave_scene_variants]
        if state.get("scene_keys") != scene_keys:
            raise ValueError("Navigation checkpoint scene identities/order do not match this training profile")
        env._exit_curriculum.load_state_dict(state["exit_curriculum"])
        # The upstream wrapper already reset at construction, before load.
        # Replace those initial spawns with the restored frontier's spawns.
        with torch.inference_mode():
            self.env.reset()
            self.alg.policy.reset()
        print(f"[INFO]: Restored navigation curriculum: {dict(zip(scene_keys, env._exit_curriculum.distance_m.tolist()))}")
        return infos


def register_navigation_runner() -> None:
    """Select the thin wrapper before Isaac Lab imports its maintained trainer.

    Isaac Lab 2.3 selects two hard-coded runner names. Rebind only the exported
    training constructor, without changing its sources or the upstream base
    class used by evaluators and policy resolution.
    """
    import rsl_rl.runners

    rsl_rl.runners.OnPolicyRunner = NavigationOnPolicyRunner
