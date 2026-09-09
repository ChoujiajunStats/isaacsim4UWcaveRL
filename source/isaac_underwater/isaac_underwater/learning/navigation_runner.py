"""Keep reverse-curriculum state in the same checkpoint as upstream PPO."""

from __future__ import annotations

import math
from pathlib import Path
import warnings

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from isaac_underwater.navigation.exit_curriculum import rehearsal_settings


class NavigationOnPolicyRunner(OnPolicyRunner):
    """Training state and explicit warm starts; optimization is upstream RSL-RL."""

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
                "rehearsal": rehearsal_settings(getattr(env, "cfg", None)),
            }
            if getattr(self, "_navigation_warm_start", None) is not None:
                infos["navigation_training"]["warm_start"] = self._navigation_warm_start
        super().save(path, infos)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        config = getattr(self, "cfg", {})
        warm_start = bool(config.get("navigation_weights_only", False))
        noise_std = config.get("navigation_reset_noise_std")
        if noise_std == 0.0:
            noise_std = None  # Disabled sentinel for Isaac Lab's typed config updater.
        if noise_std is not None and (not warm_start or not math.isfinite(noise_std) or noise_std <= 0):
            raise ValueError("Resetting exploration std requires a weights-only warm start and positive finite std")
        if warm_start:
            load_optimizer = False
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        env = self._navigation_env()
        if warm_start:
            if env is None:
                raise ValueError("Navigation weights-only warm start requires a training curriculum")
            # New optimization run: preserve model/normalizers, not optimizer,
            # old iteration number or the old curriculum's promotion state.
            self.current_learning_iteration = 0
            if noise_std is not None:
                policy = self.alg.policy
                if getattr(policy, "state_dependent_std", False):
                    raise ValueError("Noise reset supports only state-independent PPO std")
                with torch.no_grad():
                    if policy.noise_std_type == "scalar":
                        policy.std.fill_(noise_std)
                    elif policy.noise_std_type == "log":
                        policy.log_std.fill_(math.log(noise_std))
                    else:
                        raise ValueError("Unsupported PPO noise std type")
            self._navigation_warm_start = {"checkpoint": str(Path(path).resolve()),
                                           "optimizer_and_curriculum_reset": True,
                                           "noise_std_reset": noise_std}
            with torch.inference_mode():
                self.env.reset()
                self.alg.policy.reset()
            print(f"[INFO]: Navigation WEIGHTS-ONLY warm start: {self._navigation_warm_start}")
            return infos
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
        saved_rehearsal = state.get("rehearsal", rehearsal_settings(None))
        if saved_rehearsal != rehearsal_settings(getattr(env, "cfg", None)):
            raise ValueError("Navigation rehearsal config mismatch; use an explicit weights-only warm start to change it")
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
