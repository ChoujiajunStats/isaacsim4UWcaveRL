#!/usr/bin/env python3
"""Register project tasks and delegate to Isaac Lab's maintained RSL-RL trainer."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_ROOT = PROJECT_ROOT / ".deps" / "IsaacLab"
TRAIN_SCRIPT = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"

sys.path.insert(0, str(PROJECT_ROOT / "source" / "isaac_underwater"))
sys.path.insert(0, str(TRAIN_SCRIPT.parent))

import isaac_underwater.tasks  # noqa: E402,F401
from isaac_underwater.learning import register_rsl_rl_extensions  # noqa: E402

register_rsl_rl_extensions()


if __name__ == "__main__":
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")
