#!/usr/bin/env python3
"""Register project tasks and delegate to Isaac Lab's maintained RSL-RL trainer."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = PROJECT_ROOT / ".deps" / "IsaacLab"
TRAIN_SCRIPT = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"


def _consume_local_option(name: str) -> str | None:
    """Remove one project-only option before Isaac Lab parses argv."""
    prefix = name + "="
    for index, value in enumerate(tuple(sys.argv[1:]), start=1):
        if value.startswith(prefix):
            del sys.argv[index]
            return value[len(prefix) :]
        if value == name:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{name} requires a value")
            result = sys.argv[index + 1]
            del sys.argv[index : index + 2]
            return result
    return None


def _consume_local_flag(name: str) -> bool:
    if name not in sys.argv[1:]:
        return False
    sys.argv.remove(name)
    return True


cave_profile = _consume_local_option("--cave_dataset_profile")
if cave_profile:
    os.environ["ISAAC_UNDERWATER_CAVE_PROFILE"] = cave_profile
if _consume_local_flag("--domain_randomization"):
    os.environ["ISAAC_UNDERWATER_MULTICAVE_DOMAIN_RANDOMIZATION"] = "1"
if _consume_local_flag("--navigation_curriculum"):
    # Training-only Hydra override: a later evaluator process retains the
    # entrance spawn unless explicitly configured otherwise.
    sys.argv.append("env.navigation_curriculum_enabled=True")

sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))
sys.path.insert(0, str(TRAIN_SCRIPT.parent))

import isaac_underwater.tasks  # noqa: E402,F401
from isaac_underwater.learning import register_rsl_rl_extensions  # noqa: E402

register_rsl_rl_extensions()


if __name__ == "__main__":
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
