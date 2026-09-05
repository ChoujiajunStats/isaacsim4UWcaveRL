#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${project_root}/.venv/bin/python"

if [[ ! -x "${python_bin}" ]]; then
    echo "Missing ${python_bin}; run scripts/setup_isaaclab.sh first." >&2
    exit 1
fi

"${python_bin}" - <<'PY'
import importlib.metadata as metadata
import sys

print(f"Python: {sys.version.split()[0]}")
for dist in ("isaacsim", "torch", "torchvision", "rsl-rl-lib"):
    try:
        print(f"{dist}: {metadata.version(dist)}")
    except metadata.PackageNotFoundError:
        print(f"{dist}: NOT INSTALLED")

import torch
print(f"Torch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"CUDA free/total: {free / 2**20:.0f}/{total / 2**20:.0f} MiB")
PY

