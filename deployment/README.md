# Unitree Go2 Emergent Gait Locomotion

A reinforcement learning project for **Unitree Go2 locomotion**, covering the complete workflow from **IsaacLab training** to **MuJoCo Sim-to-Sim deployment**.

本项目围绕 Unitree Go2 四足机器人强化学习运动控制展开，复现并扩展 Adaptive Energy Regularization 方法，通过能量正则化引导机器人在没有显式步态先验的情况下自主学习不同速度下的高效运动方式。项目进一步完成了从 IsaacLab / RSL-RL 训练、策略导出，到 Unitree SDK2、CycloneDDS 和 MuJoCo Sim-to-Sim 部署的完整流程，主要面向四足机器人强化学习、步态涌现以及策略部署的学习与实验。

---

## Research Basis

本项目的算法主要基于以下工作进行复现与扩展：

> **Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion**  
> Boyuan Liang, Lingfeng Sun, Xinghao Zhu, Bike Zhang, Ziyin Xiong, Yixiao Wang, Chenran Li, Koushil Sreenath, Masayoshi Tomizuka  
> IEEE International Conference on Robotics and Automation (**ICRA 2025**)

原论文提出 **Adaptive Energy Regularization**，将运动跟踪与单位运动量对应的能量消耗共同纳入强化学习目标。与直接指定步态周期、接触序列或 gait command 的方法不同，该方法不显式告诉机器人应该采用 walk、trot 或其他步态，而是通过速度跟踪与能量优化，让步态作为强化学习优化过程中的结果自然涌现。

原论文主要在 **Unitree Go1** 上进行了仿真和实机验证。本项目在此基础上将训练框架迁移到 **Unitree Go2 + IsaacLab + RSL-RL**，并进一步实现 ONNX 策略导出以及独立 MuJoCo 环境中的 Sim-to-Sim 部署。

---

## Overview

当前主要训练任务为：

```text
Unitree-Go2-Adaptive-Energy-Flat
```

该平地训练任务的详细设计——包括奖励函数公式与速度课程机制——见 [平地训练任务介绍](<doc/平地训练任务介绍.md>)。

项目不仅关注策略能否训练成功，也重点关注训练与部署之间 Observation、Action、Joint Order、PD Gains 和 Control Frequency 等接口是否严格一致，使训练得到的策略可以在独立物理仿真器中重新构建完整控制闭环。

---

## Features

当前项目已经实现：

- Unitree Go2 强化学习运动控制；
- Adaptive Energy Regularization；
- 无显式 gait schedule 的步态涌现；
- IsaacLab + RSL-RL PPO 训练；
- 30 帧历史观测策略；
- PyTorch Checkpoint 与 ONNX 策略导出；
- C++ Go2 RL Controller；
- Unitree SDK2 / CycloneDDS 通信；
- LowState / LowCmd 控制闭环；
- MuJoCo Sim-to-Sim；
- FSM 状态切换；
- Nintendo Switch Pro Controller 控制。

---

## Policy

当前示例策略使用 30 帧历史 Observation。单帧 Observation 为 45 维：

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

因此策略输入为：

```text
45 × 30 = 1350
```

Actor 网络结构为：

```text
1350
 ↓
512 + ELU
 ↓
256 + ELU
 ↓
128 + ELU
 ↓
12 Actions
```

部署端将网络输出转换为关节位置目标：

```text
q_des = q_default + 0.25 × action
```

当前经过 Sim-to-Sim 验证的控制参数为：

```text
Kp      = 25.0
Kd      = 0.5
step_dt = 0.02 s
```

即策略控制频率为 **50 Hz**。

---

## Pretrained Policy

仓库提供一个已经完成 Sim-to-Sim 验证的示例策略：

```text
pretrained/example/
├── exported/
│   ├── policy.onnx
│   └── policy.onnx.data
│
└── params/
    └── deploy.yaml
```

因此第一次使用本项目时不需要立即重新训练，可以先通过该模型理解和验证完整的部署流程，再进行自己的强化学习实验。

---

## Quick Start

### Training

首先安装并激活 IsaacLab 环境，然后进入项目：

```bash
conda activate isaaclab

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

建议第一次运行时先使用较少环境验证：

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-Go2-Adaptive-Energy-Flat \
  --num_envs 32 \
  --max_iterations 10 \
  --device cuda:0
```

训练结果保存在：

```text
logs/rsl_rl/unitree_go2_adaptive_energy_flat/
```

详细安装流程请参考：

[Installation Guide](doc/Installation.md)

---

## Sim-to-Sim

Sim-to-Sim 使用独立的 MuJoCo 环境重新执行 IsaacLab 中训练得到的策略，其目标并不只是确认 ONNX 可以运行，而是检查整个 Policy Interface 是否能够在另一个物理仿真环境中保持一致。

运行时通常使用三个终端：

### Terminal 1 — Switch Bridge

```bash
cd sim2sim/switch_bridge

sudo python3 switch_to_js.py
```

### Terminal 2 — MuJoCo

```bash
cd ~/workspace/papers/unitree_mujoco/simulate/build

./unitree_mujoco -r go2 -s scene.xml
```

### Terminal 3 — RL Controller

```bash
cd deploy/robots/go2/build

./go2_ctrl --network lo
```

Controller 与 MuJoCo 连接后，机器人首先进入 `Passive` 状态，再通过 FSM 切换：

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

退出 RL 控制：

```text
Velocity
   │
   │ ZL + B
   ▼
Passive
```

---

## Documentation

训练任务本身的设计说明请先阅读：

[平地训练任务介绍](<doc/平地训练任务介绍.md>)——详细说明 Adaptive Energy 奖励函数（含公式）与速度课程设置。

完整 Sim-to-Sim 教程按照从原理到部署的顺序划分为七个阶段：

| Stage | 内容 |
|---|---|
| 0 | [什么是 Sim-to-Sim](<doc/Sim2Sim/阶段零：什么是 Sim-to-Sim.md>) |
| 1 | [IsaacLab 训练与策略导出](<doc/Sim2Sim/阶段一：IsaacLab 训练与策略导出.md>) |
| 2 | [部署接口一致性检查](<doc/Sim2Sim/阶段二：部署接口一致性检查.md>) |
| 3 | [MuJoCo 目标仿真环境搭建](<doc/Sim2Sim/阶段三：MuJoCo 目标仿真环境搭建.md>) |
| 4 | [DDS 通信链路搭建与 LowState 验证](<doc/Sim2Sim/阶段四：DDS 通信链路搭建与 LowState 验证.md>) |
| 5 | [Go2 Controller 接入 RL Policy](<doc/Sim2Sim/阶段五：Go2 Controller 接入 RL Policy.md>) |
| 6 | [FSM、Switch 手柄与完整闭环](<doc/Sim2Sim/阶段六：FSM、Switch 手柄与完整闭环.md>) |

推荐按照 Stage 0 → Stage 6 顺序阅读。

---

## Repository Structure

```text
Unitree-Go2-Emergent-Gait-Locomotion/
│
├── source/
│   └── unitree_rl_lab/
│
├── scripts/
│   └── rsl_rl/
│
├── deploy/
│   ├── include/
│   ├── robots/go2/
│   └── thirdparty/
│
├── pretrained/
│   └── example/
│
├── sim2sim/
│   ├── switch_bridge/
│   ├── dds_test/
│   └── unitree_mujoco/
│
├── doc/
│   ├── Installation.md
│   ├── 平地训练任务介绍.md
│   ├── Sim2Sim/
│   └── licenses/
│
├── NOTICE.md
├── LICENCE
├── pyproject.toml
└── unitree_rl_lab.sh
```

其中 `source/` 和 `scripts/` 主要负责 IsaacLab 强化学习训练，`deploy/` 负责 C++ RL Controller 与 ONNX 部署，`sim2sim/` 保存 MuJoCo 部署所需的辅助工具与配置，`pretrained/` 提供已经验证的示例策略，而 `doc/` 保存完整学习与复现文档。

---

## Project Status

目前已经完成：

- [x] Unitree Go2 IsaacLab environment
- [x] Adaptive Energy locomotion task
- [x] PPO training with RSL-RL
- [x] Emergent gait locomotion
- [x] Policy checkpoint and ONNX export
- [x] Deployment interface validation
- [x] Unitree SDK2 / CycloneDDS
- [x] MuJoCo Go2 simulation
- [x] LowState / LowCmd communication
- [x] C++ RL Controller
- [x] FSM
- [x] Switch Pro Controller
- [x] Complete IsaacLab → MuJoCo Sim-to-Sim

后续计划包括复杂地形训练、更加系统的步态与能耗定量评估，以及 Unitree Go2 Sim-to-Real 实机部署。

---

## Citation

本项目的 Adaptive Energy Regularization 方法来源于以下工作。如果本项目对你的研究有所帮助，请优先引用原论文：

```bibtex
@inproceedings{liang2025adaptive,
  title     = {Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion},
  author    = {Liang, Boyuan and Sun, Lingfeng and Zhu, Xinghao and Zhang, Bike
               and Xiong, Ziyin and Wang, Yixiao and Li, Chenran
               and Sreenath, Koushil and Tomizuka, Masayoshi},
  booktitle = {2025 IEEE International Conference on Robotics and Automation (ICRA)},
  pages     = {5350--5356},
  year      = {2025},
  doi       = {10.1109/ICRA55743.2025.11128812}
}
```

---

## Acknowledgements

本项目是在多个优秀开源项目与研究工作的基础上完成的，主要包括：

- Unitree Robotics `unitree_rl_lab`
- Unitree Robotics `unitree_mujoco`
- Unitree Robotics `unitree_sdk2`
- Unitree Robotics `unitree_model`
- NVIDIA IsaacLab / Isaac Sim
- RSL-RL
- MuJoCo
- ONNX Runtime
- Eclipse CycloneDDS

本仓库保留上游项目原有的版权与许可证声明。本项目属于研究复现与扩展项目，并非 Unitree Robotics、NVIDIA 或原论文作者维护的官方实现。

原始 `unitree_rl_lab` README 保存在：

```text
doc/README_unitree_rl_lab_upstream.md
```

详细第三方许可信息请参考：

```text
LICENCE
NOTICE.md
doc/licenses/
```

---
