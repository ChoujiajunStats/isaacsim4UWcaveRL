from .rsl_rl_ppo_cfg import UnderwaterPointNavPPORunnerCfg
from .visual_ppo_cfg import (
    UnderwaterCaveEntryPPORunnerCfg,
    UnderwaterExplorePPORunnerCfg,
    UnderwaterVisualPPORunnerCfg,
)

__all__ = [
    "UnderwaterCaveEntryPPORunnerCfg",
    "UnderwaterExplorePPORunnerCfg",
    "UnderwaterPointNavPPORunnerCfg",
    "UnderwaterVisualPPORunnerCfg",
    "make_runner_cfg",
    "task_requires_cameras",
]


def task_requires_cameras(task: str) -> bool:
    return "VisualPilot" in task or "Cave-Explore" in task or "Cave-Entry" in task


def make_runner_cfg(task: str):
    if "Cave-Entry" in task:
        return UnderwaterCaveEntryPPORunnerCfg()
    if "Cave-Explore" in task:
        return UnderwaterExplorePPORunnerCfg()
    if "VisualPilot" in task:
        return UnderwaterVisualPPORunnerCfg()
    return UnderwaterPointNavPPORunnerCfg()
