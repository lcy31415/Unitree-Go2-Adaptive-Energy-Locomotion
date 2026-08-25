
这一阶段的目标是：

> **让 `go2_ctrl` 正确连接 MuJoCo 仿真机器人，并加载阶段一导出的 `policy.onnx`，建立从 `LowState` 到 `LowCmd` 的完整策略推理链路。**

阶段五完成后，系统的数据流变为：

```text
MuJoCo Go2
    │
    │ rt/lowstate
    ▼
CycloneDDS
    │
    ▼
go2_ctrl
    │
    ├── LowState
    │      ↓
    ├── BaseArticulation
    │      ↓
    ├── Observation Manager
    │      ↓
    ├── 1350-dim Observation
    │      ↓
    ├── ONNX Runtime
    │      ↓
    ├── policy.onnx
    │      ↓
    ├── 12-dim Action
    │      ↓
    ├── Action Manager
    │      ↓
    └── 12 × q_des
           │
           ▼
        LowCmd
           │
           │ rt/lowcmd
           ▼
      CycloneDDS
           │
           ▼
      MuJoCo Go2
```

---

## 5.1 `go2_ctrl` 的作用

`go2_ctrl` 来自：

```text
unitree_rl_lab/deploy/robots/go2/
```

它不是 RL 模型本身，而是一个 **C++ 部署控制程序**。

可以把职责理解为：

```text
policy.onnx
= 决定“机器人应该怎么动”

go2_ctrl
= 负责“怎么获取机器人状态、运行策略，
       再把策略输出发送给机器人”
```

它主要完成：

```text
接收 LowState
      ↓
构建训练时一致的 Observation
      ↓
维护历史 Observation
      ↓
调用 ONNX Runtime
      ↓
获得 Action
      ↓
转换为 q_des
      ↓
生成并发送 LowCmd
```

---

# 5.2 DDS Domain 对齐

阶段四中，`unitree_mujoco` 使用：

```yaml
domain_id: 1
interface: "lo"
```

因此 Controller 必须处于同一个 DDS Domain。原始 `go2_ctrl/main.cpp` 使用：

```cpp
ChannelFactory::Instance()->Init(
    0,
    vm["network"].as<std::string>()
);
```

这里：

```text
0
```

表示 DDS Domain 0。

但 MuJoCo 在：

```text
Domain 1
```

因此二者无法正常发现。

修改为：

```cpp
ChannelFactory::Instance()->Init(
    1,
    vm["network"].as<std::string>()
);
```

最终：

```text
unitree_mujoco
├── Domain = 1
└── Interface = lo

go2_ctrl
├── Domain = 1
└── Interface = lo
```

形成：

```text
        Domain 1 / lo

unitree_mujoco
      │
      │ LowState
      ▼
 CycloneDDS
      │
      ▼
   go2_ctrl

unitree_mujoco
      ▲
      │
 CycloneDDS
      ▲
      │ LowCmd
   go2_ctrl
```

启动 Controller 后成功出现：

```text
Waiting for connection to robot...
Connected to robot.
```

这说明：

> `go2_ctrl` 已经能够通过 SDK2 + CycloneDDS 发现并连接 MuJoCo 中模拟的 Go2。

---

# 5.3 指定正确的 Policy Directory

官方 `go2_ctrl` 默认面向：

```text
unitree_go2_velocity
```

而本项目训练的是：

```text
Unitree-Go2-Adaptive-Energy-Flat
```

对应日志目录：

```text
logs/rsl_rl/
└── unitree_go2_adaptive_energy_flat/
```

因此修改：

```text
deploy/robots/go2/config/config.yaml
```

中的：

```yaml
policy_dir:
```

使 Controller 指向：

```text
../../../logs/rsl_rl/unitree_go2_adaptive_energy_flat
```

启动后 Controller 最终打印：

```text
Policy directory:
.../unitree_go2_adaptive_energy_flat/
2026-08-21_16-54-34_resume
```

这个 run 中包含：

```text
2026-08-21_16-54-34_resume/
├── model_9999.pt
│
├── exported/
│   ├── policy.onnx
│   └── policy.onnx.data
│
└── params/
    └── deploy.yaml
```

因此可以确认：

> Controller 使用的是由 `model_9999.pt` 导出的 Adaptive Energy Policy。

---

# 5.4 Controller 如何加载 Policy

`go2_ctrl` 并不会读取：

```text
model_9999.pt
```

它实际加载的是：

```text
exported/policy.onnx
```

流程为：

```text
config.yaml
    ↓
policy_dir
    ↓
找到对应 run
    ↓
exported/policy.onnx
    ↓
ONNX Runtime
    ↓
OrtRunner
```

也就是说：

```text
训练阶段：

model_9999.pt
      ↓
play.py
      ↓
policy.onnx


部署阶段：

go2_ctrl
      ↓
ONNX Runtime
      ↓
policy.onnx
```

训练环境和部署环境由此解耦。

---

# 5.5 编译 `go2_ctrl`

进入：

```bash
cd /home/lcy/workspace/papers/unitree_rl_lab/deploy/robots/go2
```

重新建立 build：

```bash
rm -rf build
mkdir build
cd build
```

配置：

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/unitree_robotics
```

编译：

```bash
make -j4
```

最终：

```text
[100%] Built target go2_ctrl
```

生成：

```text
deploy/robots/go2/build/go2_ctrl
```

---

# 5.6 启动 Controller

确保 `unitree_mujoco` 已经运行之后：

```bash
cd /home/lcy/workspace/papers/unitree_rl_lab/deploy/robots/go2/build

./go2_ctrl --network lo
```

成功输出：

```text
selected interface "lo" is not multicast-capable:
disabling multicast

Waiting for connection to robot...
Connected to robot.

Initializing State_Passive ...
Initializing State_FixStand ...
Initializing State_Velocity ...

Policy directory:
.../unitree_go2_adaptive_energy_flat/
2026-08-21_16-54-34_resume

FSM: Start Passive
```

其中：

```text
Connected to robot.
```

表示：

```text
DDS通信成功
LowState成功接收
```

而：

```text
Policy directory: ...
```

表示：

```text
策略目录成功定位
ONNX部署环境成功初始化
```

---

# 5.7 Controller 内部完整数据流

Controller 收到 `LowState` 后，首先由：

```text
BaseArticulation
```

解析机器人状态，例如：

```text
LowState
├── IMU gyroscope
├── IMU quaternion
├── motor q
└── motor dq
```

转换成：

```text
root_ang_vel_b
projected_gravity
joint_pos
joint_vel
```

随后：

```text
Observation Manager
```

按照：

```text
deploy.yaml
```

构造：

```text
base_ang_vel
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel
last_action
```

得到：

```text
45维 Observation
```

再堆叠：

```text
30 frames
```

得到：

```text
45 × 30
=
1350
```

于是：

```text
LowState
   ↓
BaseArticulation
   ↓
45-dim Observation
   ↓
History Buffer × 30
   ↓
1350-dim Observation
   ↓
ONNX Runtime
   ↓
policy.onnx
   ↓
12-dim Action
```

---

# 5.8 从 Action 到 LowCmd

Policy 输出：

```text
12 Actions
```

仍然不是直接发送给电机。

Action Manager 根据：

```text
deploy.yaml
```

执行：

# [  
q_{des}

q_{default}+0.25a  
]

得到：

```text
12 × q_des
```

随后按照：

```text
joint_ids_map
```

转换成 Unitree SDK motor 顺序，并写入：

```text
LowCmd
```

同时设置：

```text
Kp = 25.0
Kd = 0.5
```

最终：

```text
policy.onnx
      ↓
Action
      ↓
Action Manager
      ↓
q_des
      ↓
joint_ids_map
      ↓
LowCmd
      ↓
rt/lowcmd
      ↓
CycloneDDS
      ↓
unitree_mujoco
      ↓
MuJoCo Go2
```

到这里，策略推理闭环已经建立。

---

# 5.9 以后如何替换新的 Policy

这一架构的一个重要优势是：

> 如果新的模型保持相同的 Observation / Action 接口，通常不需要修改或重新编译 `go2_ctrl`。

未来新模型只需要：

```text
训练
 ↓
model_xxxx.pt
 ↓
play.py
 ↓
policy.onnx
policy.onnx.data
deploy.yaml
 ↓
检查部署接口
 ↓
切换 policy directory
 ↓
重新启动 go2_ctrl
```

例如新模型仍然满足：

```text
Input       = 1350
Output      = 12
History     = 30
Action      = JointPositionAction
Scale       = 0.25
Kp/Kd       = 25 / 0.5
step_dt     = 0.02
```

那么：

```text
go2_ctrl
```

可以直接复用。

真正需要重新修改 Controller 的情况主要是：

```text
Observation内容改变
Action定义改变
输出维度改变
机器人关节结构改变
新的传感器无法从LowState构造
```

因此 `go2_ctrl` 可以长期作为一个固定的部署框架，而：

```text
policy.onnx
```

作为可以不断替换的策略模块。

---
