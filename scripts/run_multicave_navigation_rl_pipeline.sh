#!/usr/bin/env bash
set -euo pipefail

# Multi-cave asset -> semantic smoke -> shared PPO -> checkpoint-load gate.
# Defaults prove the wiring only; they do not produce a converged navigator.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
TASK="${TASK:-Isaac-Underwater-Cave-Navigation-v0}"
DEVICE="${DEVICE:-cuda:0}"
CONVERT_DEVICE="${CONVERT_DEVICE:-${DEVICE}}"
PROFILE="${PROFILE:-train_all}"
NUM_ENVS="${NUM_ENVS:-6}"
SMOKE_ENVS="${SMOKE_ENVS:-3}"
EVAL_ENVS="${EVAL_ENVS:-3}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
RUN_NAME="${RUN_NAME:-multicave_pipeline}"
DOMAIN_RANDOMIZATION="${DOMAIN_RANDOMIZATION:-0}"
NAVIGATION_CURRICULUM="${NAVIGATION_CURRICULUM:-0}"
DATASET_CONFIG="${DATASET_CONFIG:-worlds/caves_difficulty_v01.yaml}"
EXPERIMENT_ROOT="${PROJECT_ROOT}/logs/rsl_rl/underwater_cave_multinav"

if [[ ! "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]] || (( NUM_ENVS % 3 != 0 )); then
  echo "NUM_ENVS must be a positive multiple of 3 for the recurrent PPO minibatches" >&2
  exit 2
fi
if [[ ! "${SMOKE_ENVS}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${EVAL_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SMOKE_ENVS and EVAL_ENVS must be positive integers" >&2
  exit 2
fi
if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

DOMAIN_ARGS=()
case "${DOMAIN_RANDOMIZATION,,}" in
  1|true|yes|on)
    DOMAIN_ARGS+=(--domain_randomization)
    ;;
  0|false|no|off)
    ;;
  *)
    echo "DOMAIN_RANDOMIZATION must be 0/1 or false/true" >&2
    exit 2
    ;;
esac

TRAIN_ARGS=()
case "${NAVIGATION_CURRICULUM,,}" in
  1|true|yes|on) TRAIN_ARGS+=(--navigation_curriculum) ;;
  0|false|no|off) ;;
  *) echo "NAVIGATION_CURRICULUM must be 0/1 or false/true" >&2; exit 2 ;;
esac

cd "${PROJECT_ROOT}"

PYTHONPATH="${PROJECT_ROOT}/source/isaac_underwater${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" scripts/test_cave_dataset.py

"${PYTHON_BIN}" scripts/convert_cave_asset.py \
  --config "${DATASET_CONFIG}" \
  --profile "${PROFILE}" \
  --headless \
  --device "${CONVERT_DEVICE}"

"${PYTHON_BIN}" scripts/smoke_cave_navigation.py \
  --num_envs "${SMOKE_ENVS}" \
  --profile "${PROFILE}" \
  "${DOMAIN_ARGS[@]}" \
  --headless \
  --device "${DEVICE}"

"${PYTHON_BIN}" scripts/train.py \
  --task "${TASK}" \
  --cave_dataset_profile "${PROFILE}" \
  "${DOMAIN_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --run_name "${RUN_NAME}" \
  --headless \
  --enable_cameras \
  --device "${DEVICE}"

RUN_DIR="$(find "${EXPERIMENT_ROOT}" -mindepth 1 -maxdepth 1 -type d \
  -name "*_${RUN_NAME}" -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "No run directory ending in _${RUN_NAME} was produced under ${EXPERIMENT_ROOT}" >&2
  exit 1
fi
CHECKPOINT="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "No checkpoint was produced in ${RUN_DIR}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/smoke_ppo.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --cave_dataset_profile "${PROFILE}" \
  "${DOMAIN_ARGS[@]}" \
  --num_envs "${EVAL_ENVS}" \
  --steps 4 \
  --headless \
  --device "${DEVICE}"

# Two seconds per episode keeps this a finite evaluator/schema contract.  Use
# the full task horizon and >=30 episodes per scene for policy acceptance.
METRICS="${RUN_DIR}/evaluation_contract.json"
"${PYTHON_BIN}" scripts/evaluate_ppo.py \
  --task "${TASK}" \
  --checkpoint "${CHECKPOINT}" \
  --cave_dataset_profile "${PROFILE}" \
  "${DOMAIN_ARGS[@]}" \
  --num_envs "${EVAL_ENVS}" \
  --episodes_per_scene 1 \
  --episode_length_s 2 \
  --max_rollout_steps 100 \
  --output "${METRICS}" \
  --headless \
  --device "${DEVICE}"

echo "multi-cave navigation RL pipeline complete (wiring gate only)"
echo "profile=${PROFILE}"
echo "checkpoint=${CHECKPOINT}"
echo "contract_metrics=${METRICS}"
