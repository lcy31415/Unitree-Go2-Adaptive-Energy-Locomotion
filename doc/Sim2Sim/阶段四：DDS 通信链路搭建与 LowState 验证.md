
这一阶段的目标是：

> **先验证 MuJoCo 中的 Go2 状态能否通过 Unitree SDK2 + CycloneDDS 正确发送给外部程序，再考虑发送控制命令。**

这里需要先区分三个东西：

```text
unitree_mujoco
= 决定仿真端使用哪个 domain / 网卡，并产生 LowState

unitree_sdk2
= 提供 LowState / LowCmd 类型和 DDS 通信接口

CycloneDDS
= 真正执行底层消息传输
```

因此，`domain_id: 1` **不是 MuJoCo 物理引擎的概念**，也不是 `unitree_mujoco` 自己实现的一套通信协议。

它本质上是：

> **DDS 的 Domain ID。**

只是由 `unitree_mujoco` 的配置文件指定，然后通过 **Unitree SDK2** 传给底层 **CycloneDDS** 使用。

---

## 4.1 仿真专用 DDS Domain

`unitree_mujoco` 中配置：

```yaml
domain_id: 1
interface: "lo"
```

对应关系是：

```text
unitree_mujoco/config.yaml
        │
        ├── domain_id = 1
        └── interface = lo
                │
                ▼
        Unitree SDK2
                │
                ▼
        ChannelFactory
                │
                ▼
          CycloneDDS
```

所以从“配置在哪里”来说：

```text
domain_id = 1
```

来自：

```text
unitree_mujoco 的 config.yaml
```

但从“它属于什么机制”来说：

```text
DDS Domain ID
```

属于 DDS/CycloneDDS 通信系统。

### `domain_id: 1`

DDS Domain 可以理解成一个**逻辑通信空间**。

只有处于相同 Domain 的节点才能互相发现和通信：

```text
MuJoCo
domain = 1
      │
      │ 可以通信
      ▼
go2_ctrl
domain = 1
```

而：

```text
MuJoCo
domain = 1

Real Go2
domain = 0
```

通常彼此隔离。

因此本项目使用：

```text
Simulation → Domain 1
```

可以避免仿真消息与未来真机通信混在一起。

---

## 4.2 `interface: lo`

`lo` 是 Linux 的：

```text
loopback interface
```

也就是：

```text
127.0.0.1
```

对应的本机网络接口。

因此：

```yaml
interface: "lo"
```

表示 DDS 通信只通过本机 loopback 进行。

整个链路变成：

```text
┌────────────── 本机 ──────────────┐
│                                  │
│ unitree_mujoco                   │
│       │                          │
│       │ LowState                 │
│       ▼                          │
│ CycloneDDS                       │
│       │                          │
│       │ lo                       │
│       ▼                          │
│ go2_ctrl / DDS test              │
│                                  │
└──────────────────────────────────┘
```

此时不需要通过 Wi-Fi 或网线寻找机器人。

所以：

```text
domain_id = 1
```

解决的是：

> **“属于哪个 DDS 通信空间？”**

而：

```text
interface = lo
```

解决的是：

> **“通过哪张网络接口通信？”**

这两个概念不要混淆。

---

# 4.3 `rt/lowstate` 是怎么产生的

启动 `unitree_mujoco` 后：

```text
MuJoCo Go2
    │
    ├── joint position q
    ├── joint velocity dq
    ├── IMU quaternion
    ├── gyroscope
    └── ...
    │
    ▼
unitree_mujoco
    │
    │ 填入 SDK2 LowState 数据结构
    ▼
LowState
    │
    │ publish
    ▼
rt/lowstate
```

这里的职责分别是：

```text
MuJoCo
→ 产生物理状态

unitree_mujoco
→ 将 MuJoCo 状态转换成 Unitree LowState

unitree_sdk2
→ 提供 LowState 类型和 Publisher

CycloneDDS
→ 把 LowState 消息发送出去
```

因此：

> **`rt/lowstate` 的实际数据生产者是 `unitree_mujoco`，通信接口来自 Unitree SDK2，底层传输由 CycloneDDS 完成。**

---

# 4.4 为什么先只读 LowState

这一阶段我们没有直接发送：

```text
rt/lowcmd
```

而是先写一个只读测试程序：

```text
dds_read_go2
```

只做：

```text
Subscribe
    ↓
rt/lowstate
```

测试结构非常简单：

```text
unitree_mujoco
      │
      │ publish LowState
      ▼
 rt/lowstate
      │
      ▼
 CycloneDDS
      │
      ▼
 unitree_sdk2 Subscriber
      │
      ▼
 dds_read_go2
```

它不会对机器人发送任何命令，所以即使测试代码有问题，也不会导致机器人突然运动。

---

# 4.5 测试程序实际上验证了什么

测试程序同样初始化：

```text
DDS Domain = 1
Network Interface = lo
```

然后通过 Unitree SDK2 订阅：

```text
rt/lowstate
```

最终能够持续读取：

```text
FR_hip q = -0.12946
gyro ≈ 0
```

这里：

```text
FR_hip q
```

来自：

```text
MuJoCo FR_hip joint
        ↓
unitree_mujoco
        ↓
LowState.motor_state
        ↓
rt/lowstate
```

而：

```text
gyro
```

来自：

```text
MuJoCo IMU
   ↓
unitree_mujoco
   ↓
LowState.imu_state
   ↓
rt/lowstate
```

因此这并不仅仅证明“程序没有报错”。

它实际上同时证明了：

```text
MuJoCo状态产生             ✓
        ↓
unitree_mujoco状态转换      ✓
        ↓
LowState数据结构            ✓
        ↓
SDK2 Publisher              ✓
        ↓
rt/lowstate Topic           ✓
        ↓
CycloneDDS                  ✓
        ↓
Domain 1                    ✓
        ↓
lo网络接口                   ✓
        ↓
SDK2 Subscriber             ✓
        ↓
外部C++程序读取             ✓
```

---

# 4.6 SDK2 在这一阶段起什么作用

这一阶段特别能体现 SDK2 的作用。

如果没有 SDK2，你需要自己完成：

```text
定义 LowState 数据格式
        ↓
编写 DDS IDL
        ↓
生成 DDS C++ 类型
        ↓
创建 Publisher
        ↓
创建 Subscriber
        ↓
处理 CycloneDDS 初始化
```

而 Unitree SDK2 已经提供：

```text
LowState
LowCmd

ChannelFactory

Publisher
Subscriber

Go2 DDS message definitions
```

所以程序只需要告诉 SDK2：

```text
使用 Domain 1
使用 lo
订阅 rt/lowstate
```

剩下的数据传输由：

```text
SDK2
 ↓
CycloneDDS
```

完成。

因此这里可以把 SDK2 理解成：

> **Go2 控制程序与 CycloneDDS 之间的机器人专用通信接口层。**

---

# 4.7 三者关系最容易这样理解

```text
             unitree_mujoco
                    │
                    │ 读取 MuJoCo 状态
                    ▼
                LowState
                    │
          ┌─────────┴─────────┐
          │   Unitree SDK2    │
          │                   │
          │ Message Type      │
          │ Publisher         │
          │ ChannelFactory    │
          └─────────┬─────────┘
                    │
                    ▼
               CycloneDDS
                    │
              Domain ID = 1
              Interface = lo
                    │
                    ▼
          ┌───────────────────┐
          │   Unitree SDK2    │
          │    Subscriber     │
          └─────────┬─────────┘
                    │
                    ▼
             dds_read_go2
```

---
# 4.8 为什么使用 Domain 1 很重要

后续真机部署时，一个很重要的安全原则是不要让仿真 DDS 和真实机器人 DDS 混在一起。

推荐保持：

```text
Simulation
├── domain = 1
└── interface = lo

Real Go2
├── domain = 0
└── interface = 实际机器人网卡
```

这样即使：

```text
rt/lowstate
rt/lowcmd
```

Topic 名称完全相同，它们仍然属于两个不同通信空间。

可以理解为：

```text
Domain 0
┌─────────────────────────┐
│ Real Go2                 │
│ rt/lowstate              │
│ rt/lowcmd                │
└─────────────────────────┘


Domain 1
┌─────────────────────────┐
│ MuJoCo Go2               │
│ rt/lowstate              │
│ rt/lowcmd                │
└─────────────────────────┘
```

这对后续 Sim-to-Real 是一个很好的安全隔离设计。

---
