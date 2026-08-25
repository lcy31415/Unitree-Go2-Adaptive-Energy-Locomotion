## 0.1 Sim-to-Sim 是什么

**Sim-to-Sim（Simulation-to-Simulation）** 是指将在一个仿真器中训练得到的机器人控制策略，迁移到另一个独立仿真器中运行和验证。

在本项目中：

```text
Source Simulator：IsaacLab / Isaac Sim
Target Simulator：MuJoCo
Robot：Unitree Go2
Policy：RSL-RL / PPO 训练得到的神经网络策略
```

整体过程可以表示为：

```text
IsaacLab
   │
   │ PPO Training
   ▼
model_xxxx.pt
   │
   │ Export
   ▼
policy.onnx
   │
   │ Deployment
   ▼
Go2 Controller
   │
   │ LowCmd
   ▼
MuJoCo Go2
   │
   │ LowState
   └──────────────→ Controller
```

与普通的 IsaacLab `play.py` 不同，Sim-to-Sim 阶段中 **IsaacLab 不再负责机器人动力学仿真**。策略从 IsaacLab 中导出后，由独立的 C++ Controller 加载，并根据 MuJoCo 返回的机器人状态进行推理，再将控制指令发送回 MuJoCo，从而形成完整闭环。

---

## 0.2 为什么需要 Sim-to-Sim

强化学习策略通常是在某一个特定仿真环境中训练得到的，例如：

```text
IsaacLab
```

但不同物理引擎对于机器人动力学的实现并不完全相同。例如 Isaac Sim/PhysX 与 MuJoCo 在以下方面可能存在差异：

```text
接触模型
摩擦模型
碰撞求解
关节约束
积分器
执行器模型
时间离散
数值误差
```

因此，一个策略能够在 IsaacLab 中稳定运行，并不意味着它一定真正学习到了具有泛化能力的机器人运动控制规律。而如果能够达到：

```text
IsaacLab
   ✓

MuJoCo
   ✓
```

则说明策略至少能够跨越一定程度的 **Simulator Gap**。因此，Sim-to-Sim 可以作为 Sim-to-Real 之前非常重要的一层验证。

---
# 0.3 Sim-to-Sim 的核心不是“转换模型格式”

一个常见误解是：

```text
model.pt
   ↓
policy.onnx
```

完成之后就是 Sim-to-Sim。实际上，**ONNX 导出只是整个过程中的很小一部分**。真正的核心是：

> 在 Target Simulator 中重新构建与训练阶段完全一致的 Policy Interface。

即：

```text
训练阶段                           部署阶段

IsaacLab Observation       ==     MuJoCo Observation

Observation Scale          ==     Observation Scale

Joint Order                ==     Joint Order

Coordinate Frame           ==     Coordinate Frame

Action Definition          ==     Action Definition

Action Scale               ==     Action Scale

Default Joint Position     ==     Default Joint Position

PD Controller              ≈      PD Controller

Control Frequency          ==     Control Frequency
```

只有这些条件基本一致，神经网络看到的输入才具有和训练阶段相同的物理含义。

---

# 0.4 本项目中的 Sim-to-Sim 架构

本项目采用以下架构：
![[微信图片_20260822005924_14_2.jpg]]
---

# 0.5 本项目中策略的数据流

本项目中的 Go2 Policy 使用 **30 帧历史观测**。

单帧 Observation：

```text
base angular velocity        3
projected gravity            3
velocity command             3
joint position              12
joint velocity              12
previous action             12
──────────────────────────────
Total                        45
```

因此 Policy 输入为：

```text
45 × 30 = 1350
```

实际 ONNX 网络：

```text
Observation
[1 × 1350]
      │
      ▼
Linear
1350 → 512
      │
     ELU
      │
      ▼
Linear
512 → 256
      │
     ELU
      │
      ▼
Linear
256 → 128
      │
     ELU
      │
      ▼
Linear
128 → 12
      │
      ▼
Action
[1 × 12]
```

最终 12 个 Action 对应 Go2 的 12 个驱动关节。

---

# 0.6 为什么需要 Unitree SDK2 和 DDS

本项目没有让 ONNX Policy 直接调用 MuJoCo API。

而是复用了 Unitree 的机器人通信接口：

```text
Go2 Controller
       ↕
Unitree SDK2
       ↕
CycloneDDS
       ↕
unitree_mujoco
```

主要通信 Topic 为：

```text
rt/lowstate
rt/lowcmd
```

其中：

```text
rt/lowstate
```

负责：

```text
MuJoCo → Controller
```

发送：

```text
IMU
joint position
joint velocity
joystick state
...
```

而：

```text
rt/lowcmd
```

负责：

```text
Controller → MuJoCo
```

发送：

```text
q_des
dq_des
Kp
Kd
torque
```

这样做有一个非常大的优势：

> MuJoCo 仿真和真实 Unitree Go2 可以共享非常相似的 Controller 接口。

因此未来从：

```text
MuJoCo
```

切换到：

```text
Real Go2
```

时，上层 Policy/Controller 的结构可以尽量保持不变。
