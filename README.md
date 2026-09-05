# Isaac Sim Underwater PPO

A low-VRAM, headless Isaac Lab research environment for vectorized underwater
navigation on an NVIDIA RTX 4060 8 GB GPU.

The first milestone is `Isaac-Underwater-PointNav-Direct-v0`: a lightweight
4-DOF rigid body controlled by body-frame surge, sway, heave force, and yaw
torque. Training uses state observations and RSL-RL PPO; cameras, RTX rendering,
ROS, and complex meshes are intentionally excluded.

## Pinned stack

- Isaac Sim 5.1.0 (pip distribution)
- Isaac Lab v2.3.2 (`37ddf626871758333d6ed89cf64ad702aef127d0`)
- Python 3.11
- PyTorch 2.7.0 + CUDA 12.8 runtime
- RSL-RL 3.1.2

See [docs/environment.md](docs/environment.md) for the measured workstation
baseline and compatibility rationale. Setup, training, evaluation, and measured
benchmark commands will be added as each executable milestone is validated.

