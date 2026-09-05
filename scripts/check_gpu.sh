#!/usr/bin/env bash
set -euo pipefail

echo "NVIDIA GPU summary"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version --format=csv

echo
echo "Active GPU processes"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true

echo
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
else
    echo "nvcc: not installed (not required by the pinned pip installation)"
fi

