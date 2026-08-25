
这一阶段的目标是：

> **把手柄输入、FSM 状态切换、RL Policy、DDS 通信和 MuJoCo 机器人连接成一个完整可控的 Sim-to-Sim 闭环。**

前五个阶段已经完成：

```text
MuJoCo Go2
   ↕
LowState / LowCmd
   ↕
CycloneDDS
   ↕
go2_ctrl
   ↕
policy.onnx
```

阶段六再加入：

```text
Switch Controller
```

最终系统变成：

```text
Switch Controller
      │
      ▼
 /dev/input/js0
      │
      ▼
unitree_mujoco
      │
      │ joystick state
      ▼
   LowState
      │
      │ rt/lowstate
      ▼
 CycloneDDS
      │
      ▼
   go2_ctrl
      │
      ├── FSM
      │
      └── RL Policy
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
             │
             └────────→ 新 LowState
```

---

## 6.1 为什么需要 FSM

RL Policy 不应该在程序启动的一瞬间直接接管机器人。

因为机器人刚启动时可能处于：

```text
趴姿
关节未对齐
策略历史观测为空
```

如果立即执行 RL Action，很容易产生较大的关节跳变。

因此 `go2_ctrl` 使用 FSM：

```text
Finite State Machine
有限状态机
```

对控制过程进行分阶段管理。

本项目主要使用三个状态：

```text
Passive
   ↓
FixStand
   ↓
Velocity / RL
```

---

## 6.2 Passive：安全初始状态

Controller 启动后首先进入：

```text
FSM: Start Passive
```

此时 RL Policy 不直接控制机器人运动。

可以理解为：

> **机器人已连接，但尚未进入主动运动状态。**

系统此时主要完成：

```text
建立 DDS 通信
接收 LowState
初始化机器人状态
等待用户指令
```

因此 `Passive` 是整个控制流程的安全起点。

---

## 6.3 FixStand：从当前姿态过渡到站立姿态

通过手柄：

```text
ZL + A
```

触发：

```text
Passive
   ↓
FixStand
```

`FixStand` 的作用不是运行 RL，而是通过预先设定的关节轨迹，使机器人平滑进入标准站立姿态。

大致过程：

```text
当前关节位置
      ↓
插值轨迹
      ↓
目标站立姿态
```

这样可以避免：

```text
趴姿
 ↓
直接交给RL
 ↓
关节突然跳变
```

因此推荐始终采用：

```text
Passive
   ↓
FixStand
   ↓
RL
```

而不是直接：

```text
Passive → RL
```

---

## 6.4 Velocity：RL Policy 接管

机器人完成站立后，按：

```text
+
```

进入：

```text
Velocity
```

也就是当前部署的 RL 控制状态。

此时：

```text
LowState
   ↓
Observation
   ↓
30-frame History
   ↓
1350-dim Input
   ↓
policy.onnx
   ↓
12-dim Action
   ↓
q_des
   ↓
LowCmd
```

开始持续运行。

策略控制周期：

```text
step_dt = 0.02 s
```

因此约为：

```text
50 Hz
```

也就是说每 20 ms：

```text
读取状态
  ↓
执行一次策略
  ↓
生成新的关节目标
```

---

# 6.5 Switch Pro Controller 接入

本项目使用：

```text
Nintendo Switch Pro Controller
```

通过蓝牙连接 Ubuntu。

Linux 能正确识别：

```text
hid_nintendo
```

并产生：

```text
/dev/input/eventXX
```

但当时没有自动生成：

```text
/dev/input/js0
```

而 `unitree_mujoco` 的 joystick 模块需要标准 Linux joystick 接口：

```text
/dev/input/js0
```

因此增加了一层转换。

---

# 6.6 `switch_to_js.py`

我们建立了：

```text
unitree_mujoco/
└── tools/
    └── switch_bridge/
        └── switch_to_js.py
```

作用是：

```text
Switch Pro Controller
       ↓
Linux evdev
       ↓
/dev/input/eventXX
       ↓
switch_to_js.py
       ↓
Virtual Switch Pro Controller
       ↓
/dev/input/js0
```

也就是说，它把：

```text
evdev input device
```

转换成：

```text
Linux joystick device
```

供 `unitree_mujoco` 使用。

启动：

```bash
cd /home/lcy/workspace/papers/unitree_mujoco/tools/switch_bridge

sudo python3 switch_to_js.py
```

该终端需要持续运行。

---

# 6.7 验证 `/dev/input/js0`

Bridge 启动后，应出现：

```text
/dev/input/js0
```

可以使用：

```bash
jstest /dev/input/js0
```

检查按键和摇杆。

本项目实测得到：

```text
Axes:
0  Left Stick X
1  Left Stick Y
2  Right Stick X
3  Right Stick Y
4  D-pad X
5  D-pad Y
```

主要按键映射为：

```text
A   → button 1
B   → button 0
X   → button 3
Y   → button 2

L   → button 5
R   → button 6

ZL  → button 7
ZR  → button 8

-   → button 9
+   → button 10
```

这一步非常重要，因为：

> Switch 手柄上的物理按键名称，不一定与 Linux joystick 的 button index 一致。

因此应该先用 `jstest` 实测，再修改程序映射。

---

# 6.8 `unitree_mujoco` 中的 Switch 映射

对应修改位于：

```text
unitree_mujoco/simulate/src/physics_joystick.h
```

最终主要映射为：

```cpp
back(js_->button_[9]);       // -

start(js_->button_[10]);     // +

LB(js_->button_[5]);         // L
RB(js_->button_[6]);         // R

A(js_->button_[1]);
B(js_->button_[0]);
X(js_->button_[3]);
Y(js_->button_[2]);

LT(js_->button_[7]);         // ZL
RT(js_->button_[8]);         // ZR
```

方向键：

```cpp
up(js_->axis_[5] < 0);
down(js_->axis_[5] > 0);

left(js_->axis_[4] < 0);
right(js_->axis_[4] > 0);
```

摇杆：

```cpp
lx(double(js_->axis_[0]) / max_value_);
ly(-double(js_->axis_[1]) / max_value_);

rx(double(js_->axis_[2]) / max_value_);
ry(-double(js_->axis_[3]) / max_value_);
```

因此：

```text
Switch输入
   ↓
Linux js0
   ↓
SwitchJoystick
   ↓
Unitree remote state
   ↓
LowState
   ↓
go2_ctrl
```

---

# 6.9 `unitree_mujoco` joystick 配置

仿真配置：

```text
unitree_mujoco/simulate/config.yaml
```

最终使用：

```yaml
robot: "go2"
robot_scene: "scene.xml"

domain_id: 1
interface: "lo"

use_joystick: 1
joystick_type: "switch"
joystick_device: "/dev/input/js0"

joystick_bits: 16
```

其中：

```text
use_joystick: 1
```

表示开启 joystick。

```text
joystick_type: switch
```

表示使用 Switch 映射。

```text
joystick_device: /dev/input/js0
```

指定实际输入设备。

---

# 6.10 FSM 按键控制

当前 FSM 主要使用：

```text
ZL + A
```

执行：

```text
Passive
   ↓
FixStand
```

随后：

```text
+
```

执行：

```text
FixStand
   ↓
Velocity / RL
```

紧急退出 / 返回 Passive：

```text
ZL + B
```

所以完整操作流程：

```text
启动系统
   ↓
Passive
   ↓
ZL + A
   ↓
FixStand
   ↓
等待站稳
   ↓
+
   ↓
Velocity / RL
```

需要退出策略时：

```text
ZL + B
   ↓
Passive
```

---

# 6.11 摇杆如何控制 RL

进入：

```text
Velocity
```

之后，手柄不再直接控制关节。

而是产生：

```text
velocity_commands
```

作为 Policy Observation 的一部分。

逻辑为：

```text
Left Stick
    ↓
目标平移速度

Right Stick
    ↓
目标转向速度
```

然后：

```text
velocity command
      │
      ▼
Observation
      │
      ▼
policy.onnx
      │
      ▼
12 actions
```

因此手柄真正控制的是：

> **“机器人想往哪里运动”**

而不是：

> **“某一个关节应该转多少度”**

这是 locomotion policy 的典型控制方式。

---

# 6.12 完整 Sim-to-Sim 启动顺序

最终稳定运行需要三个终端。

### Terminal 1：Switch Bridge

```bash
cd /home/lcy/workspace/papers/unitree_mujoco/tools/switch_bridge

sudo python3 switch_to_js.py
```

作用：

```text
Switch Controller
       ↓
/dev/input/js0
```

---

### Terminal 2：MuJoCo Go2

```bash
cd /home/lcy/workspace/papers/unitree_mujoco/simulate/build

./unitree_mujoco -r go2 -s scene.xml
```

作用：

```text
MuJoCo Dynamics
+
Go2
+
LowState / LowCmd
+
Joystick
```

---

### Terminal 3：RL Controller

```bash
cd /home/lcy/workspace/papers/unitree_rl_lab/deploy/robots/go2/build

./go2_ctrl --network lo
```

成功后应该看到类似：

```text
Waiting for connection to robot...
Connected to robot.

Initializing State_Passive ...
Initializing State_FixStand ...
Initializing State_Velocity ...

FSM: Start Passive
```

然后：

```text
ZL + A
   ↓
FixStand

+
   ↓
Velocity / RL
```

---

# 6.13 完整闭环到底发生了什么

进入 RL 后，每一个策略周期：

```text
① MuJoCo计算机器人状态
          ↓
② unitree_mujoco生成LowState
          ↓
③ rt/lowstate
          ↓
④ CycloneDDS
          ↓
⑤ go2_ctrl接收LowState
          ↓
⑥ 构造45维Observation
          ↓
⑦ 堆叠30帧 → 1350维
          ↓
⑧ policy.onnx推理
          ↓
⑨ 得到12维Action
          ↓
⑩ q_des = q_default + 0.25 × action
          ↓
⑪ 写入LowCmd
          ↓
⑫ rt/lowcmd
          ↓
⑬ CycloneDDS
          ↓
⑭ unitree_mujoco接收LowCmd
          ↓
⑮ PD Controller
          ↓
⑯ MuJoCo产生新的机器人状态
          ↓
             回到①
```

同时，Switch 手柄提供：

```text
FSM command
+
velocity command
```

从而形成完整的人机闭环。
