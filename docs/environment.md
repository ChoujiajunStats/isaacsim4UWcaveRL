# Environment and compatibility record

Measured on 2026-09-05 before installation.

| Item | Value |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS (x86_64) |
| Kernel | 7.0.0-30-generic |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8,188 MiB |
| NVIDIA driver | 595.84 (open kernel module package) |
| Driver maximum CUDA API | 13.2 |
| System CUDA toolkit | Not installed (`nvcc` absent) |
| RAM | 15 GiB total, 10 GiB available at inspection |
| Swap | 14 GiB |
| Root disk | 923 GiB total, 403 GiB available |
| System Python | 3.12.3 |

The desktop session used about 1.7 GiB VRAM during inspection. Benchmark output
therefore records both total board usage and process-local PyTorch allocation;
long training should be run without other GPU-heavy desktop applications.

## Selected stack

| Component | Pin |
| --- | --- |
| Isaac Sim | 5.1.0, official pip packages |
| Isaac Lab | v2.3.2 / commit `37ddf626871758333d6ed89cf64ad702aef127d0` |
| Python | 3.11 |
| PyTorch | 2.7.0 |
| Torch CUDA runtime | cu128 (CUDA 12.8) |
| RSL-RL | 3.1.2 (`rsl-rl-lib`) |
| Environment manager | uv 0.12.5, project-local `.venv` |

Isaac Lab v2.3.2 explicitly supports Isaac Sim 4.5/5.0/5.1 and identifies
5.1.0 as its current release target. NVIDIA's Isaac Sim 5.1 requirements list
Ubuntu 22.04/24.04 and Linux driver 580.65.06; the installed 595.84 driver is
newer and remains untouched. Isaac Sim 5.x requires Python 3.11. The absence of
system `nvcc` is expected for the pip workflow because Isaac Sim and PyTorch
ship the needed CUDA runtime libraries.

Official references:

- https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html
- https://github.com/isaac-sim/IsaacLab/tree/v2.3.2
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html

