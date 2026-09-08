from .rsl_rl_ppo_cfg import UnderwaterPointNavPPORunnerCfg
from .visual_ppo_cfg import (
    UnderwaterCaveEntryPPORunnerCfg,
    UnderwaterExplorePPORunnerCfg,
    UnderwaterMultiCaveNavigationPPORunnerCfg,
    UnderwaterVisualPPORunnerCfg,
)

__all__ = [
    "UnderwaterCaveEntryPPORunnerCfg",
    "UnderwaterExplorePPORunnerCfg",
    "UnderwaterMultiCaveNavigationPPORunnerCfg",
    "UnderwaterPointNavPPORunnerCfg",
    "UnderwaterVisualPPORunnerCfg",
    "make_runner_cfg",
    "task_requires_cameras",
]


def task_requires_cameras(task: str) -> bool:
    return any(
        marker in task
        for marker in ("VisualPilot", "Cave-Explore", "Cave-Entry", "Cave-Navigation")
    )


def make_runner_cfg(task: str):
    if "Cave-Navigation" in task:
        return UnderwaterMultiCaveNavigationPPORunnerCfg()
    if "Cave-Entry" in task:
        return UnderwaterCaveEntryPPORunnerCfg()
    if "Cave-Explore" in task:
        return UnderwaterExplorePPORunnerCfg()
    if "VisualPilot" in task:
        return UnderwaterVisualPPORunnerCfg()
    return UnderwaterPointNavPPORunnerCfg()
