#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

with_perception=0
if [[ "${1:-}" == "--with-perception" ]]; then
    with_perception=1
    shift
fi

for num_envs in 32 64 128 256 512; do
    "${project_root}/.venv/bin/python" scripts/benchmark.py \
        --mode physics --num_envs "${num_envs}" --headless --device cuda:0 \
        --output logs/benchmarks/physics.csv "$@"
done

if [[ "${with_perception}" == "1" ]]; then
    for num_envs in 1 4 8 16; do
        "${project_root}/.venv/bin/python" scripts/benchmark.py \
            --mode perception --num_envs "${num_envs}" --headless --device cuda:0 \
            --warmup_steps 20 --measure_steps 100 \
            --output logs/benchmarks/perception.csv "$@"
    done
else
    echo "Perception sweep skipped; pass --with-perception after fixing the host RTX camera startup." >&2
fi
