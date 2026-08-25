
这一阶段的目标是：

> **先建立一个能够独立运行 Go2 的 MuJoCo 仿真环境，并确认机器人模型、动力学和 Viewer 均正常工作，再接入 DDS 与强化学习控制器。**

此时暂时不运行 `policy.onnx`，也不启动 `go2_ctrl`。首先只解决：

```text
MuJoCo
   ↓
加载 Go2
   ↓
执行动力学
   ↓
Viewer 正常显示
```

这样后续出现问题时，可以明确区分是 **MuJoCo 环境问题**，还是 **DDS / Controller / Policy 问题**。

---

## 3.1 项目目录设计

本项目采用：

```text
~/workspace/robotics/
└── MuJoCo/
    └── mujoco-3.3.6/

~/workspace/papers/
└── unitree_mujoco/
```

两个目录职责不同：

```text
MuJoCo 3.3.6
= 通用物理仿真器

unitree_mujoco
= Unitree 机器人 MuJoCo 仿真项目
```

即：

```text
unitree_mujoco
      │
      │ 调用
      ▼
MuJoCo 3.3.6
      │
      ▼
Go2 Dynamics
```

这种方式可以避免把公共软件与具体实验项目混在一起。

---

## 3.2 MuJoCo 3.3.6

MuJoCo 是本项目 Sim-to-Sim 的 **Target Simulator**。

训练阶段使用：

```text
IsaacLab / Isaac Sim / PhysX
```

而部署验证阶段改为：

```text
MuJoCo 3.3.6
```

因此：

```text
IsaacLab
   ↓
训练 Policy

MuJoCo
   ↓
独立验证 Policy
```

`unitree_mujoco` 编译时不仅需要 MuJoCo 动态库，还需要完整发行包中的：

```text
mujoco-3.3.6/
├── include/
│   └── mujoco/
├── lib/
│   └── libmujoco.so
├── simulate/
│   ├── glfw_adapter.h
│   ├── glfw_adapter.cc
│   └── ...
└── ...
```

特别是：

```text
simulate/glfw_adapter.h
```

`unitree_mujoco` 会直接使用 MuJoCo Viewer/Simulate 中的部分源码，因此只安装一个系统 `libmujoco` 并不一定足够。

---

## 3.3 `unitree_mujoco`

项目目录：

```text
~/workspace/papers/unitree_mujoco
```

其中 C++ 仿真器位于：

```text
unitree_mujoco/
└── simulate/
    ├── src/
    ├── config.yaml
    ├── CMakeLists.txt
    ├── mujoco -> ...
    └── build/
```

它的作用可以概括为：

```text
MuJoCo
   ↓
Go2物理模型
   ↓
unitree_mujoco
   ↓
模拟 Unitree Go2 底层接口
```

后续它还负责产生：

```text
LowState
```

并接收：

```text
LowCmd
```

但在阶段三中，我们暂时只验证它能否正确运行 MuJoCo Go2。

---

## 3.4 建立 MuJoCo 软链接

为了避免在 `unitree_mujoco` 项目内部复制一整套 MuJoCo，我们使用软链接：

```text
unitree_mujoco/simulate/mujoco
                   │
                   │ symbolic link
                   ▼
~/workspace/robotics/MuJoCo/mujoco-3.3.6
```

对应：

```bash
cd /home/lcy/workspace/papers/unitree_mujoco/simulate

ln -s \
/home/lcy/workspace/robotics/MuJoCo/mujoco-3.3.6 \
mujoco
```

这样：

```text
simulate/mujoco/include
simulate/mujoco/lib
simulate/mujoco/simulate
```

实际上都会访问公共的 MuJoCo 安装目录。

可以检查：

```bash
readlink -f \
/home/lcy/workspace/papers/unitree_mujoco/simulate/mujoco
```

正确结果应指向：

```text
/home/lcy/workspace/robotics/MuJoCo/mujoco-3.3.6
```

---

## 3.5 检查 MuJoCo 依赖是否完整

在编译前建议检查三个关键文件：

```bash
ls \
/home/lcy/workspace/robotics/MuJoCo/mujoco-3.3.6/include/mujoco/mujoco.h

ls \
/home/lcy/workspace/robotics/MuJoCo/mujoco-3.3.6/simulate/glfw_adapter.h

ls -lh \
/home/lcy/workspace/robotics/MuJoCo/mujoco-3.3.6/lib/libmujoco.so*
```

它们分别负责：

|文件|作用|
|---|---|
|`mujoco.h`|MuJoCo C/C++ API|
|`glfw_adapter.h`|MuJoCo Viewer/GLFW 图形适配|
|`libmujoco.so`|MuJoCo 核心动态库|

三者都存在后再继续编译。

---

## 3.6 编译 `unitree_mujoco`

依赖准备完成后：

```bash
cd /home/lcy/workspace/papers/unitree_mujoco/simulate

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

其中：

```text
/opt/unitree_robotics
```

是 Unitree SDK2 的安装位置。

然后：

```bash
make -j4
```

最终成功：

```text
[100%] Linking CXX executable unitree_mujoco
[100%] Built target unitree_mujoco
```

并生成：

```text
simulate/build/
├── unitree_mujoco
└── jstest
```

其中：

```text
unitree_mujoco
= Go2 MuJoCo仿真程序

jstest
= 手柄输入测试程序
```

---

## 3.7 启动 Go2 MuJoCo 仿真

编译完成后，首先只启动 Target Simulator：

```bash
cd /home/lcy/workspace/papers/unitree_mujoco/simulate/build

./unitree_mujoco \
  -r go2 \
  -s scene.xml
```

参数含义：

```text
-r go2
   ↓
选择 Unitree Go2

-s scene.xml
   ↓
选择 MuJoCo 场景
```

此时应该正常弹出：

```text
MuJoCo Viewer
```

并看到：

```text
Go2
+
Ground / Terrain
+
MuJoCo Scene
```

这一阶段机器人可能处于趴姿，这是正常现象，因为：

```text
MuJoCo Simulator 已启动
```

但：

```text
go2_ctrl 尚未启动
RL Policy 尚未接管
```

因此还没有控制器给机器人发送站立动作。
