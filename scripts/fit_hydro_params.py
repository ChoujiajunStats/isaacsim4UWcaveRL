#!/usr/bin/env python3
"""Calibration entry point; requires an external reference trajectory.

The script intentionally refuses to fit guessed values without measured or
reference data. It validates the input contract and reports the next step.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True, help="Unified experimental/reference trajectory CSV")
    args = parser.parse_args()
    with args.reference.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Reference trajectory is empty")
    required = {"timestamp", "x", "y", "z", "u", "v", "w", "p", "q", "r"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Reference trajectory missing columns: {sorted(missing)}")
    print(f"validated_rows={len(rows)}")
    print("calibration: NOT YET RUN; fit only after reference replay and unit/frame checks")


if __name__ == "__main__":
    main()
