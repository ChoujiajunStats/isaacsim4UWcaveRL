#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_root}/.venv"
isaaclab_dir="${project_root}/.deps/IsaacLab"

# NVIDIA's package index can be slow for multi-hundred-MiB wheels.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-10}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }

retry() {
    local attempt
    for attempt in 1 2 3; do
        "$@" && return 0
        echo "Command failed (attempt ${attempt}/3); retrying..." >&2
        sleep 3
    done
    return 1
}

if [[ ! -x "${venv_dir}/bin/python" ]]; then
    uv venv --python 3.11 --seed "${venv_dir}"
fi

if [[ ! -d "${isaaclab_dir}/.git" ]]; then
    mkdir -p "${project_root}/.deps"
    retry git -c http.version=HTTP/1.1 clone --branch v2.3.2 --depth 1 \
        --filter=blob:none --sparse \
        https://github.com/isaac-sim/IsaacLab.git "${isaaclab_dir}"
    git -C "${isaaclab_dir}" sparse-checkout set source scripts apps tools
fi

# Install the pinned CUDA build first so Isaac Sim reuses it instead of
# downloading the default PyPI torch wheel and replacing it afterward.
uv pip install --python "${venv_dir}/bin/python" --upgrade \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "${venv_dir}/bin/python" \
    "isaacsim[all,extscache]==5.1.0" \
    --extra-index-url https://pypi.nvidia.com

# flatdict 4.0.1 imports pkg_resources from its legacy setup.py.  Recent
# isolated build environments omit it, so keep a compatible setuptools in the
# project environment and build flatdict without isolation before Isaac Lab.
uv pip install --python "${venv_dir}/bin/python" 'setuptools<81'
uv pip install --python "${venv_dir}/bin/python" --no-build-isolation flatdict==4.0.1

TERM=xterm-256color VIRTUAL_ENV="${venv_dir}" PATH="${venv_dir}/bin:${PATH}" \
    "${isaaclab_dir}/isaaclab.sh" --install rsl_rl

# Isaac Lab's optional tooling otherwise upgrades packages that Isaac Sim 5.1
# pins strictly.  Older ONNX/IPython/wheel releases satisfy both stacks.
uv pip install --python "${venv_dir}/bin/python" \
    psutil==5.9.8 typing-extensions==4.12.2 \
    'ipython<9' onnx==1.18.0 wheel==0.43.0

echo "Isaac Sim and Isaac Lab installation complete."
