#!/usr/bin/env python3
"""Import a Stonefish trajectory into the project unified CSV contract.

Stonefish is intentionally not launched here.  The importer makes future
independent-simulator replay possible without treating Stonefish as ground
truth.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REQUIRED_COLUMNS = (
    "timestamp", "x", "y", "z", "qw", "qx", "qy", "qz",
    "u", "v", "w", "p", "q", "r", "current_x", "current_y", "current_z",
)


def load_unified_trajectory(path: Path) -> list[dict[str, float | str]]:
    """Read and validate a unified trajectory CSV exported by Stonefish."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Missing unified trajectory columns: {', '.join(missing)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            parsed: dict[str, float | str] = {"experiment": row.get("experiment", "stonefish")}
            for column in REQUIRED_COLUMNS:
                try:
                    value = float(row[column])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid {column} at line {line_number}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {column} at line {line_number}")
                parsed[column] = value
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Trajectory is empty: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Unified Stonefish trajectory CSV")
    parser.add_argument("--output", type=Path, default=None, help="Optional normalized CSV output")
    args = parser.parse_args()
    if args.input is None:
        print("stonefish_compare: NOT AVAILABLE (no Stonefish trajectory supplied)")
        print("stonefish_trajectory_importer: PASS")
        return
    rows = load_unified_trajectory(args.input)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["experiment", *REQUIRED_COLUMNS]
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(f"stonefish_rows={len(rows)} importer=PASS")
    if args.output is not None:
        print(f"normalized_output={args.output}")


if __name__ == "__main__":
    main()
