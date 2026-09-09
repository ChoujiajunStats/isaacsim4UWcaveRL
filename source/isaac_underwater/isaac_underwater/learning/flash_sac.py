"""Minimal compatibility bridge to a pinned, unmodified upstream FlashSAC.

Do not run upstream's uv sync inside the Isaac workspace: its Python 3.11
configuration upgrades torch and installs several unrelated simulators.
This adapter retains the existing torch stack and only supplies NumPy/torch
type aliases instead of importing JAX solely for a Union annotation.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any, Union

import numpy as np
import numpy.typing as npt
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
FLASHSAC_COMMIT = "87edc9061150ae9e962dd84e6544e27a1554b3ab"


def prepare_flashsac() -> Path:
    repo = PROJECT_ROOT / ".deps/FlashSAC"
    if not (repo / "flash_rl/agents/flashSAC/agent.py").is_file():
        raise FileNotFoundError(f"Clone https://github.com/Holiday-Robot/FlashSAC into {repo}")
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != FLASHSAC_COMMIT:
        raise RuntimeError(f"Expected FlashSAC revision {FLASHSAC_COMMIT}, got {revision}")
    # Also disables the upstream helper functions decorated unconditionally
    # with torch.compile, not just its configurable actor/critic compilation.
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if "flash_rl.types" not in sys.modules and importlib.util.find_spec("jax") is None:
        aliases = ModuleType("flash_rl.types", "Torch-only aliases for upstream's typing-only JAX dependency.")
        aliases.NDArray = npt.NDArray[Any]
        aliases.F32NDArray = npt.NDArray[np.float32]
        aliases.Tensor = Union[npt.NDArray[Any], torch.Tensor]
        sys.modules["flash_rl.types"] = aliases
    return repo


def load_flashsac_config(*, device: str, **overrides):
    prepare_flashsac()
    from flash_rl.agents.flashSAC.agent import FlashSACConfig

    path = PROJECT_ROOT / "configs/learning/flash_sac_caves.yaml"
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    values.update(device_type=device, buffer_device_type=device)
    values.update(overrides)
    if values["n_step"] != 1:
        raise ValueError("The cave adapter is currently validated only for one-step replay")
    if values["use_compile"] or values["use_amp"]:
        raise ValueError("The pinned-stack pilot uses eager float32; compiled/AMP performance is not validated")
    if not values["asymmetric_observation"]:
        raise ValueError("The cave actor must not receive the privileged critic columns")
    if not 0 < values["buffer_min_length"] <= values["buffer_max_length"] or values["sample_batch_size"] < 2:
        raise ValueError("Invalid replay capacity/warmup/batch size")
    return FlashSACConfig(**values)


def replay_storage_bytes(capacity: int, observation_dim: int, action_dim: int) -> int:
    # Two float32 observations, action, reward, terminated, truncated.
    return capacity * (2 * observation_dim + action_dim + 3) * 4


def pack_flashsac_observation(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    # Upstream's asymmetric critic sees actor input plus privileged state.
    # Only the first policy-width columns ever reach the actor.
    return torch.cat((observations["policy"], observations["critic"]), dim=-1)


def make_flashsac_agent(actor_dim: int, critic_dim: int, num_envs: int, cfg):
    prepare_flashsac()
    import gymnasium as gym
    from flash_rl.agents.flashSAC.agent import FlashSACAgent

    if num_envs <= 0 or cfg.buffer_max_length < num_envs:
        raise ValueError("Replay capacity must hold at least one complete vector step")
    return FlashSACAgent(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(actor_dim + critic_dim,), dtype=np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, shape=(num_envs, 6), dtype=np.float32),
        env_info={"actor_observation_size": (actor_dim,)}, cfg=cfg,
    )


def restore_flashsac_training_state(agent, checkpoint: Path, *, old_capacity: int, num_envs: int) -> None:
    """Restore one-step replay while allowing an explicitly resized vector job.

    Replay rows do not depend on worker identity. Per-worker partial reward
    returns do, so they must be re-created for the new simulator episodes.
    Growing a ring buffer requires chronological re-packing: merely loading
    upstream's old write pointer into a larger buffer exposes uninitialized
    rows to sampling as soon as its logical length increases.
    """
    if agent._cfg.n_step != 1 or num_envs <= 0 or old_capacity <= 0:
        raise ValueError("Resized training resume requires one-step replay and positive worker count")
    new_capacity = agent._cfg.buffer_max_length
    if new_capacity < old_capacity:
        raise ValueError("Training resume may grow replay but cannot silently discard it by shrinking capacity")
    if new_capacity == old_capacity:
        agent.load_replay_buffer(str(checkpoint))
    else:
        dataset = torch.load(checkpoint / "replay_buffer.pt", map_location="cpu", weights_only=True)
        count = int(dataset["num_in_buffer"])
        pointer = int(dataset["current_idx"])
        if not 0 <= count <= old_capacity or not 0 <= pointer < old_capacity:
            raise ValueError("Invalid saved replay size/write pointer")
        if count < old_capacity and pointer != count:
            raise ValueError("Invalid partially filled replay write pointer")
        fields = ("observation", "next_observation", "action", "reward", "terminated", "truncated")
        if any(dataset[key].shape[0] != count for key in fields):
            raise ValueError("Saved replay tensors have inconsistent lengths")
        order = (torch.arange(count) + pointer) % count if count == old_capacity else torch.arange(count)
        if len(agent._replay_buffer):
            raise ValueError("Load resized replay into a fresh agent to avoid overwriting live training data")
        for start in range(0, count, 2048):
            indices = order[start:start + 2048]
            agent._replay_buffer.add({key: dataset[key][indices] for key in fields})
        if len(agent._replay_buffer) != count or agent._replay_buffer._current_idx != count % new_capacity:
            raise RuntimeError("Resized replay did not preserve its valid rows")
    if agent.reward_normalizer is not None:
        agent.reward_normalizer.G_r = torch.zeros(num_envs, device=agent._device)


class TerminalObservationMixin:
    """Capture actual terminal observations before DirectRLEnv auto-reset.

    Only used by the nominal, ground-truth-localization FlashSAC adapter.
    The original PPO environment and its observation timing are unchanged.
    """

    def step(self, actions):
        self._capture_terminal_observation = True
        self._captured_terminal_ids = None
        self._captured_terminal_observation = None
        try:
            observations, rewards, terminated, truncated, extras = super().step(actions)
        finally:
            self._capture_terminal_observation = False
        final_observations = {key: value.detach().clone() for key, value in observations.items()}
        if self._captured_terminal_ids is not None:
            for key, values in self._captured_terminal_observation.items():
                final_observations[key][self._captured_terminal_ids] = values
        extras = dict(extras)
        extras["final_observation"] = final_observations
        return observations, rewards, terminated, truncated, extras

    def _reset_idx(self, env_ids):
        if getattr(self, "_capture_terminal_observation", False):
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            terminal = self._get_observations()
            self._captured_terminal_ids = ids.clone()
            self._captured_terminal_observation = {key: value[ids].detach().clone() for key, value in terminal.items()}
        super()._reset_idx(env_ids)
