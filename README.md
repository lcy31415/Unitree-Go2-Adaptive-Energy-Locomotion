# Unitree Go2 Emergent Gait Locomotion

A beginner-friendly reinforcement learning project for **Unitree Go2 locomotion**, covering the complete workflow from **IsaacLab / RSL-RL training** to **MuJoCo Sim-to-Sim deployment**.

本项目以 Unitree Go2 四足机器人为对象，提供一套从强化学习训练、策略导出到 MuJoCo Sim-to-Sim 部署的完整示例，适合作为四足机器人强化学习与部署流程的入门项目。

---

## Overview

本项目的完整流程为：

```text
IsaacLab
   ↓
RSL-RL / PPO Training
   ↓
model_xxxx.pt
   ↓
Policy Export
   ↓
policy.onnx
   ↓
Go2 Controller
   ↓
Unitree SDK2
   ↓
CycloneDDS
   ↓
unitree_mujoco
   ↓
MuJoCo Go2
```

当前主要训练任务：

```text
Unitree-Go2-Adaptive-Energy-Flat
```

项目重点包括：

- Unitree Go2 locomotion
- PPO reinforcement learning
- Emergent gait learning
- Adaptive energy reward
- IsaacLab policy training
- ONNX policy export
- Deployment interface validation
- Unitree SDK2
- CycloneDDS
- MuJoCo Sim-to-Sim
- FSM controller
- Nintendo Switch Pro Controller

---

## System Architecture

完整 Sim-to-Sim 架构：

```text
                    Switch Controller
                           │
                           ▼
                    /dev/input/js0
                           │
                           ▼
┌─────────────────────────────────────────────┐
│               unitree_mujoco               │
│                                             │
│               MuJoCo Go2                   │
│                   │                         │
│              Sensors / Joints              │
│                   │                         │
│                LowState                    │
└───────────────────┼─────────────────────────┘
                    │
               rt/lowstate
                    │
                    ▼
               CycloneDDS
                    │
                    ▼
┌─────────────────────────────────────────────┐
│                  go2_ctrl                   │
│                                             │
│ LowState                                    │
│    ↓                                        │
│ Observation Construction                    │
│    ↓                                        │
│ Observation History                         │
│    ↓                                        │
│ policy.onnx                                 │
│    ↓                                        │
│ Action                                      │
│    ↓                                        │
│ Joint Position Target                       │
│    ↓                                        │
│ LowCmd                                      │
└───────────────────┼─────────────────────────┘
                    │
                rt/lowcmd
                    │
                    ▼
               CycloneDDS
                    │
                    ▼
                MuJoCo Go2
```

其中：

| Component | Function |
|---|---|
| IsaacLab | 强化学习训练环境 |
| RSL-RL | PPO 强化学习算法 |
| `policy.onnx` | 部署使用的 Actor 网络 |
| `go2_ctrl` | C++ Go2 RL 控制器 |
| Unitree SDK2 | LowState / LowCmd 与 DDS 通信接口 |
| CycloneDDS | 状态和控制命令的数据传输 |
| `unitree_mujoco` | Unitree Go2 MuJoCo 接口仿真 |
| MuJoCo | Target Simulator / 物理引擎 |
| Switch Controller | FSM 与速度命令输入 |

---

## RL Policy

当前示例策略使用 30 帧历史观测。

单帧 Observation：

```text
base_ang_vel          3
projected_gravity     3
velocity_commands     3
joint_pos_rel        12
joint_vel_rel        12
last_action          12
───────────────────────
Total                45
```

历史长度：

```text
45 × 30 = 1350
```

因此 Actor 网络为：

```text
Input
1350
 ↓
Linear 1350 → 512
 ↓ ELU
Linear 512 → 256
 ↓ ELU
Linear 256 → 128
 ↓ ELU
Linear 128 → 12
 ↓
Output
12 Actions
```

Actor 参数量：

```text
857,484
```

部署 Action 采用 Joint Position Target：

```text
q_des = q_default + 0.25 × action
```

当前示例部署参数：

```text
Kp = 25.0
Kd = 0.5
Policy frequency = 50 Hz
```

---

## Repository Structure

```text
Unitree-Go2-Emergent-Gait-Locomotion/
│
├── source/
│   └── unitree_rl_lab/
│       └── Unitree RL environments and tasks
│
├── scripts/
│   └── rsl_rl/
│       ├── train.py
│       └── play.py
│
├── deploy/
│   ├── include/
│   ├── robots/
│   │   └── go2/
│   └── thirdparty/
│
├── pretrained/
│   └── example/
│       ├── exported/
│       │   ├── policy.onnx
│       │   └── policy.onnx.data
│       └── params/
│           └── deploy.yaml
│
├── doc/
│   ├── Sim2Sim/
│   └── licenses/
│
├── docker/
├── NOTICE.md
├── LICENCE
├── pyproject.toml
└── unitree_rl_lab.sh
```

---

# Installation

## 1. Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Unitree-Go2-Emergent-Gait-Locomotion
```

> GitHub 仓库正式创建后，将 `<YOUR_GITHUB_REPOSITORY_URL>` 替换为实际地址。

---

## 2. External Dependencies

本项目依赖：

```text
Ubuntu 22.04
NVIDIA GPU

Isaac Sim
IsaacLab
RSL-RL

Unitree SDK2
unitree_model
unitree_mujoco

MuJoCo 3.3.6
ONNX Runtime
CycloneDDS
```

推荐目录：

```text
~/workspace/
├── robotics/
│   ├── IsaacLab/
│   ├── MuJoCo/
│   │   └── mujoco-3.3.6/
│   ├── unitree_sdk2/
│   ├── unitree_model/
│   └── unitree_ros/
│
└── papers/
    ├── Unitree-Go2-Emergent-Gait-Locomotion/
    └── unitree_mujoco/
```

机器人资源路径可以通过环境变量配置：

```bash
export UNITREE_MODEL_DIR=$HOME/workspace/robotics/unitree_model
export UNITREE_ROS_DIR=$HOME/workspace/robotics/unitree_ros
```

---

# Training

激活 IsaacLab 环境：

```bash
conda activate isaaclab
```

进入项目：

```bash
cd ~/workspace/papers/Unitree-Go2-Emergent-Gait-Locomotion
```

启动训练：

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-Go2-Adaptive-Energy-Flat \
  --num_envs 4096 \
  --max_iterations 5000 \
  --device cuda:0
```

训练结果默认保存在：

```text
logs/rsl_rl/unitree_go2_adaptive_energy_flat/
```

其中：

```text
model_xxxx.pt
```

是 RSL-RL 训练 checkpoint。

---

# Policy Export

训练完成后使用 `play.py` 加载 checkpoint 并导出部署策略：

```bash
RUN=/path/to/your/training/run

python scripts/rsl_rl/play.py \
  --task Unitree-Go2-Adaptive-Energy-Flat \
  --load_run <RUN_NAME> \
  --checkpoint "$RUN/model_xxxx.pt" \
  --num_envs 1 \
  --device cuda:0
```

导出后：

```text
exported/
├── policy.pt
├── policy.onnx
└── policy.onnx.data
```

其中当前 C++ `go2_ctrl` 使用：

```text
policy.onnx
policy.onnx.data
```

---

# Pretrained Policy

仓库提供一个已经验证通过 Sim-to-Sim 的示例策略：

```text
pretrained/example/
├── exported/
│   ├── policy.onnx
│   └── policy.onnx.data
│
└── params/
    └── deploy.yaml
```

默认 `go2_ctrl` 会读取：

```text
pretrained/example
```

因此不重新训练也可以直接学习完整 Sim-to-Sim 流程。

---

# Sim-to-Sim

详细教程按照七个阶段组织：

| Stage | Tutorial |
|---|---|
| 0 | [什么是 Sim-to-Sim](<doc/Sim2Sim/阶段零：什么是 Sim-to-Sim.md>) |
| 1 | [IsaacLab 训练与策略导出](<doc/Sim2Sim/阶段一：IsaacLab 训练与策略导出.md>) |
| 2 | [部署接口一致性检查](<doc/Sim2Sim/阶段二：部署接口一致性检查.md>) |
| 3 | [MuJoCo 目标仿真环境搭建](<doc/Sim2Sim/阶段三：MuJoCo 目标仿真环境搭建.md>) |
| 4 | [DDS 通信链路搭建与 LowState 验证](<doc/Sim2Sim/阶段四：DDS 通信链路搭建与 LowState 验证.md>) |
| 5 | [Go2 Controller 接入 RL Policy](<doc/Sim2Sim/阶段五：Go2 Controller 接入 RL Policy.md>) |
| 6 | [FSM、Switch 手柄与完整闭环](<doc/Sim2Sim/阶段六：FSM、Switch 手柄与完整闭环.md>) |

建议按照：

```text
Stage 0
  ↓
Stage 1
  ↓
Stage 2
  ↓
Stage 3
  ↓
Stage 4
  ↓
Stage 5
  ↓
Stage 6
```

顺序学习。

---

# Run Sim-to-Sim

完整系统运行时需要三个终端。

## Terminal 1 — Switch Controller Bridge

```bash
cd /path/to/unitree_mujoco/tools/switch_bridge

sudo python3 switch_to_js.py
```

生成：

```text
/dev/input/js0
```

---

## Terminal 2 — MuJoCo

```bash
cd /path/to/unitree_mujoco/simulate/build

./unitree_mujoco -r go2 -s scene.xml
```

仿真通信配置：

```yaml
domain_id: 1
interface: "lo"
```

---

## Terminal 3 — Go2 RL Controller

```bash
cd deploy/robots/go2/build

./go2_ctrl --network lo
```

正常启动后：

```text
Waiting for connection to robot...
Connected to robot.
FSM: Start Passive
```

---

# FSM Control

机器人不会在启动后立即进入 RL Policy，而是通过 FSM 分阶段切换：

```text
Passive
   │
   │ ZL + A
   ▼
FixStand
   │
   │ +
   ▼
Velocity / RL
```

返回 Passive：

```text
Velocity
   │
   │ ZL + B
   ▼
Passive
```

其中：

```text
Passive
= 安全初始状态

FixStand
= 平滑进入标准站立姿态

Velocity
= RL Policy 接管机器人
```

---

# Sim-to-Sim Closed Loop

进入 `Velocity` 后，每个策略周期执行：

```text
MuJoCo State
      ↓
LowState
      ↓
rt/lowstate
      ↓
CycloneDDS
      ↓
go2_ctrl
      ↓
45-dim Observation
      ↓
30-frame History
      ↓
1350-dim Observation
      ↓
policy.onnx
      ↓
12 Actions
      ↓
q_des
      ↓
LowCmd
      ↓
rt/lowcmd
      ↓
CycloneDDS
      ↓
MuJoCo
```

策略周期：

```text
step_dt = 0.02 s
```

即约：

```text
50 Hz
```

---

# Important Deployment Notes

Sim-to-Sim 中最重要的不是只有 ONNX 输入输出维度一致，而是需要保证训练端与部署端的完整 Policy Interface 一致：

```text
Observation order
Observation scale
Coordinate frame
History order

Joint order
Action scale
Default joint position

PD gains
Control frequency
```

例如本项目曾发现自动导出的：

```yaml
stiffness: [0.0, ...]
damping: [0.0, ...]
```

会导致机器人在 `FixStand → Velocity` 后立即倒下。

当前示例策略正确使用：

```text
Kp = 25.0
Kd = 0.5
```

这也是本项目 Sim-to-Sim 教程重点讨论的问题之一。

---

# Project Status

- [x] Unitree Go2 IsaacLab environment
- [x] Adaptive Energy locomotion task
- [x] PPO training with RSL-RL
- [x] Emergent gait locomotion
- [x] Policy checkpoint
- [x] ONNX export
- [x] Deployment interface validation
- [x] MuJoCo 3.3.6
- [x] Unitree SDK2
- [x] CycloneDDS
- [x] LowState / LowCmd communication
- [x] Go2 C++ Controller
- [x] FSM
- [x] Switch Pro Controller
- [x] Complete IsaacLab → MuJoCo Sim-to-Sim

Future work:

- [ ] Quantitative Sim-to-Sim evaluation
- [ ] More emergent gait experiments
- [ ] Robust terrain locomotion
- [ ] Sim-to-Real deployment on Unitree Go2

---

# Acknowledgements

This project is based on and inspired by the following open-source projects:

- NVIDIA IsaacLab
- Unitree Robotics `unitree_rl_lab`
- Unitree Robotics `unitree_mujoco`
- Unitree Robotics `unitree_sdk2`
- Unitree Robotics `unitree_model`
- RSL-RL
- MuJoCo
- ONNX Runtime
- Eclipse CycloneDDS

This repository contains modifications and extensions for Unitree Go2 emergent gait locomotion and Sim-to-Sim deployment.

Original copyright and license notices from upstream projects are preserved.

See:

```text
LICENCE
NOTICE.md
doc/licenses/
```

for license information.

---

# Disclaimer

This repository is intended for research and educational purposes.

When transferring controllers from simulation to a real Unitree Go2, always verify joint mapping, control gains, network configuration, emergency stop behavior, and robot safety procedures before enabling low-level control.