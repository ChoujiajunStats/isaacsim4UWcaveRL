#!/usr/bin/env bash
set -euo pipefail

# Finite automatic-portal entry train -> evaluate -> checkpoint gate.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
TASK="${TASK:-Isaac-Underwater-Cave-Entry-v0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-2}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
EVAL_ENVS="${EVAL_ENVS:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
RUN_NAME="${RUN_NAME:-cave_entry_pipeline}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/train.py \
  --task "${TASK}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --run_name "${RUN_NAME}" \
  --headless \
  --enable_cameras \
  --device "${DEVICE}"

CHECKPOINT="$(find "${PROJECT_ROOT}/logs/rsl_rl/underwater_cave_entry" -maxdepth 3 -type f -name 'model_*.pt' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "No entry checkpoint was produced under logs/rsl_rl/underwater_cave_entry" >&2
  exit 1
fi

METRICS="$(dirname "${CHECKPOINT}")/evaluation.json"
"${PYTHON_BIN}" scripts/evaluate_ppo.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --num_envs "${EVAL_ENVS}" \
  --episodes "${EVAL_EPISODES}" \
  --output "${METRICS}" \
  --headless \
  --device "${DEVICE}"

"${PYTHON_BIN}" scripts/smoke_ppo.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --num_envs "${EVAL_ENVS}" \
  --steps 4 \
  --headless \
  --device "${DEVICE}"

echo "cave-entry RL pipeline complete"
echo "checkpoint=${CHECKPOINT}"
echo "metrics=${METRICS}"
