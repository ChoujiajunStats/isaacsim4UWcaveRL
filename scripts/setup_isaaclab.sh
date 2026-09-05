#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_root}/.venv"
isaaclab_dir="${project_root}/.deps/IsaacLab"

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    uv venv --python 3.11 --seed "${venv_dir}"
fi

if [[ ! -d "${isaaclab_dir}/.git" ]]; then
    mkdir -p "${project_root}/.deps"
    git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "${isaaclab_dir}"
fi

uv pip install --python "${venv_dir}/bin/python" \
    "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com
uv pip install --python "${venv_dir}/bin/python" --upgrade \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

VIRTUAL_ENV="${venv_dir}" PATH="${venv_dir}/bin:${PATH}" \
    "${isaaclab_dir}/isaaclab.sh" --install rsl_rl

echo "Isaac Sim and Isaac Lab installation complete."

