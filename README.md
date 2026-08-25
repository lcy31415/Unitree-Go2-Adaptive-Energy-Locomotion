# Unitree Go2 Adaptive-Energy Locomotion

This repository combines the complete training and deployment workflow for
adaptive-energy Unitree Go2 locomotion.

## Repository layout

- `training/`: Isaac Lab environments, adaptive-energy rewards, LP-ACRL
  curricula, training scripts, evaluation tools, and policy export.
- `deployment/`: MuJoCo sim-to-sim validation, Unitree DDS integration,
  offline policy checks, monitoring tools, and real-robot deployment.

## Main tasks

- `Unitree-Go2-Adaptive-Energy-Flat`
- `Unitree-Go2-Adaptive-Energy-Flat-LPACRL`
- `Unitree-Go2-Adaptive-Energy-Terrain-LPACRL`

Run training and evaluation commands from the `training/` directory. Refer to
the README files under `training/` and `deployment/` for environment setup and
component-specific usage.

## Generated artifacts

Training logs, checkpoints, exported policies, videos, build products, and
local backup files are intentionally excluded from Git. Release-ready policy
artifacts should be distributed separately, for example through a GitHub
Release or Git LFS.

## Upstream projects and licensing

The training framework is derived from Unitree Robotics' `unitree_rl_lab` and
the NVIDIA Isaac Lab ecosystem. Third-party license texts and notices retained
from the source projects remain under the corresponding `training/` and
`deployment/` directories. Review those notices before redistribution.
