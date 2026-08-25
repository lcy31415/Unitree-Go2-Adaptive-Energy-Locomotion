# Installation

本文档介绍 `Unitree-Go2-Emergent-Gait-Locomotion` 的推荐安装方式。

项目包含两套相互独立的运行环境：

```text
Training
└── Conda
    ├── Isaac Sim
    ├── IsaacLab
    ├── PyTorch
    └── RSL-RL

Sim-to-Sim
└── System / C++
    ├── MuJoCo
    ├── unitree_mujoco
    ├── unitree_sdk2
    ├── CycloneDDS
    └── ONNX Runtime
```

建议不要把所有依赖安装到同一个 Conda 环境中。

---

# 1. Recommended Directory Structure

推荐使用以下目录：

```text
/home/<USER>/
│
├── miniconda3/
│
└── workspace/
    │
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

原则：

```text
robotics/
→ 公共框架和机器人依赖

papers/
→ 论文复现和具体研究项目
```

---

# 2. System Requirements

推荐：

```text
Ubuntu 22.04
NVIDIA GPU
NVIDIA Driver
Git
CMake
GCC / G++
Python / Conda
```

检查 GPU：

```bash
nvidia-smi
```

检查系统：

```bash
lsb_release -a
```

检查编译器：

```bash
gcc --version
g++ --version
cmake --version
```

---

# 3. Clone This Repository

```bash
mkdir -p ~/workspace/papers
cd ~/workspace/papers

git clone <REPOSITORY_URL>

cd Unitree-Go2-Emergent-Gait-Locomotion
```

---

# 4. Training Environment

训练部分使用独立 Conda 环境。

本项目当前验证环境：

```text
Python       3.12
Isaac Sim    6.0.1
IsaacLab     3.0 beta series
PyTorch      2.11 + CUDA 12.8
RSL-RL       5.x
```

> Isaac Sim / IsaacLab 的兼容关系变化较快，建议优先参考对应版本官方安装文档。

---

## 4.1 Install Miniconda

如果已经安装 Miniconda，可以跳过。

推荐位置：

```text
~/miniconda3
```

安装完成后建议关闭 base 自动激活：

```bash
conda config --set auto_activate_base false
```

---

## 4.2 Create IsaacLab Environment

```bash
conda create -n isaaclab python=3.12 -y

conda activate isaaclab
```

确认：

```bash
which python
python --version
```

---

## 4.3 Install Isaac Sim

在 `isaaclab` 环境中安装与 IsaacLab 对应的 Isaac Sim。

安装完成后测试：

```bash
python -c "import isaacsim; print('Isaac Sim OK')"
```

---

## 4.4 Install IsaacLab

推荐将 IsaacLab 放在：

```text
~/workspace/robotics/IsaacLab
```

例如：

```bash
mkdir -p ~/workspace/robotics
cd ~/workspace/robotics

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

切换到与本项目兼容的版本后安装。

安装完成后建议首先测试官方 Cartpole 示例，确认 IsaacLab 本身能够正常工作，再安装本项目。

---

# 5. Install This Project

激活：

```bash
conda activate isaaclab
```

进入项目：

```bash
cd ~/workspace/papers/Unitree-Go2-Emergent-Gait-Locomotion
```

安装项目：

```bash
python -m pip install -e source/unitree_rl_lab
```

验证：

```bash
python -c "import unitree_rl_lab; print(unitree_rl_lab.__file__)"
```

输出路径应指向当前仓库：

```text
.../Unitree-Go2-Emergent-Gait-Locomotion/source/unitree_rl_lab/...
```

---

# 6. Unitree Robot Assets

本项目使用 Unitree Go2 机器人资源。

推荐：

```bash
cd ~/workspace/robotics

git clone https://github.com/unitreerobotics/unitree_model.git
git clone https://github.com/unitreerobotics/unitree_ros.git
```

设置：

```bash
export UNITREE_MODEL_DIR=$HOME/workspace/robotics/unitree_model
export UNITREE_ROS_DIR=$HOME/workspace/robotics/unitree_ros
```

可以写入：

```text
~/.bashrc
```

例如：

```bash
echo 'export UNITREE_MODEL_DIR=$HOME/workspace/robotics/unitree_model' >> ~/.bashrc
echo 'export UNITREE_ROS_DIR=$HOME/workspace/robotics/unitree_ros' >> ~/.bashrc

source ~/.bashrc
```

---

# 7. Verify IsaacLab Environment

查看任务：

```bash
conda activate isaaclab

cd ~/workspace/papers/Unitree-Go2-Emergent-Gait-Locomotion

python scripts/list_envs.py
```

确认能够找到：

```text
Unitree-Go2-Adaptive-Energy-Flat
```

然后先做一个小规模训练测试：

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-Go2-Adaptive-Energy-Flat \
  --num_envs 32 \
  --max_iterations 10 \
  --device cuda:0
```

如果能够正常进入 PPO rollout，则训练环境基本安装成功。

---

# 8. MuJoCo Sim-to-Sim Environment

Sim-to-Sim 部分建议退出 Conda：

```bash
conda deactivate
```

避免 Conda 的库影响 CMake 和系统 C++ 依赖。

安装基础依赖：

```bash
sudo apt update

sudo apt install -y \
  build-essential \
  cmake \
  git \
  libyaml-cpp-dev \
  libspdlog-dev \
  libboost-all-dev \
  libglfw3-dev \
  libeigen3-dev \
  libfmt-dev
```

---

# 9. Install Unitree SDK2

推荐位置：

```text
~/workspace/robotics/unitree_sdk2
```

```bash
cd ~/workspace/robotics

git clone https://github.com/unitreerobotics/unitree_sdk2.git

cd unitree_sdk2
mkdir build
cd build

cmake ..
make -j$(nproc)

sudo make install
```

本项目推荐安装到：

```text
/opt/unitree_robotics
```

构建其他 Unitree 程序时可以使用：

```bash
-DCMAKE_PREFIX_PATH=/opt/unitree_robotics
```

---

# 10. Install MuJoCo

本项目验证使用：

```text
MuJoCo 3.3.6
```

推荐目录：

```text
~/workspace/robotics/MuJoCo/mujoco-3.3.6
```

最终结构：

```text
~/workspace/robotics/MuJoCo/
└── mujoco-3.3.6/
    ├── bin/
    ├── include/
    ├── lib/
    └── sample/
```

---

# 11. Install unitree_mujoco

`unitree_mujoco` 作为外部项目使用，不直接复制进本仓库。

```bash
cd ~/workspace/papers

git clone https://github.com/unitreerobotics/unitree_mujoco.git

cd unitree_mujoco
```

将 MuJoCo 指向：

```text
~/workspace/robotics/MuJoCo/mujoco-3.3.6
```

例如：

```bash
cd simulate

ln -s \
  ~/workspace/robotics/MuJoCo/mujoco-3.3.6 \
  mujoco
```

检查：

```bash
ls -l mujoco
```

---

# 12. Apply Sim-to-Sim Configuration

本仓库提供：

```text
sim2sim/unitree_mujoco/
```

其中包含本项目验证过的配置。

核心 DDS 配置：

```yaml
domain_id: 1
interface: "lo"
```

Joystick：

```yaml
use_joystick: 1
joystick_type: "switch"
joystick_device: "/dev/input/js0"
```

将这些修改应用到外部 `unitree_mujoco` 后再构建。

---

# 13. Build unitree_mujoco

进入：

```bash
cd ~/workspace/papers/unitree_mujoco/simulate
```

构建：

```bash
mkdir -p build
cd build

cmake ..
make -j$(nproc)
```

运行：

```bash
./unitree_mujoco -r go2 -s scene.xml
```

---

# 14. Verify DDS

在启动 RL Controller 之前，建议先验证：

```text
MuJoCo
  ↓
LowState
  ↓
CycloneDDS
  ↓
SDK2
```

本仓库提供：

```text
sim2sim/dds_test/
```

构建：

```bash
cd ~/workspace/papers/Unitree-Go2-Emergent-Gait-Locomotion/sim2sim/dds_test

mkdir -p build
cd build

cmake .. -DCMAKE_PREFIX_PATH=/opt/unitree_robotics

make -j4
```

保持 MuJoCo 运行，然后启动 DDS test。

如果能够持续读取：

```text
rt/lowstate
```

说明 DDS 通信正常。

---

# 15. Build Go2 Controller

```bash
cd ~/workspace/papers/Unitree-Go2-Emergent-Gait-Locomotion/deploy/robots/go2

mkdir -p build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/unitree_robotics

make -j4
```

---

# 16. Pretrained Policy

项目提供：

```text
pretrained/example/
├── exported/
│   ├── policy.onnx
│   └── policy.onnx.data
└── params/
    └── deploy.yaml
```

因此第一次测试不需要重新训练。

当前部署控制参数：

```text
Kp = 25.0
Kd = 0.5
step_dt = 0.02 s
```

即：

```text
Policy Frequency = 50 Hz
```

---

# 17. Run Complete Sim-to-Sim

建议依次启动。

## Terminal 1 — Switch Bridge

```bash
cd ~/workspace/papers/Unitree-Go2-Emergent-Gait-Locomotion/sim2sim/switch_bridge

sudo python3 switch_to_js.py
```

确认：

```bash
ls /dev/input/js0
```

---

## Terminal 2 — MuJoCo

```bash
cd ~/workspace/papers/unitree_mujoco/simulate/build

./unitree_mujoco -r go2 -s scene.xml
```

---

## Terminal 3 — RL Controller

```bash
cd ~/workspace/papers/Unitree-Go2-Emergent-Gait-Locomotion/deploy/robots/go2/build

./go2_ctrl --network lo
```

正常情况下：

```text
Connected to robot.
FSM: Start Passive
```

然后：

```text
ZL + A
→ FixStand

+
→ Velocity / RL
```

---

# 18. Troubleshooting

## Robot falls immediately after entering RL

首先检查：

```text
pretrained/example/params/deploy.yaml
```

确保不是：

```yaml
stiffness:
  - 0.0

damping:
  - 0.0
```

当前策略使用：

```text
Kp = 25.0
Kd = 0.5
```

---

## Joystick open failed

如果出现：

```text
Joystick open failed
```

检查：

```bash
ls /dev/input/js0
```

如果不存在，先启动：

```bash
sudo python3 sim2sim/switch_bridge/switch_to_js.py
```

---

## Controller cannot receive LowState

确认三者一致：

```text
unitree_mujoco:
Domain ID = 1
Interface = lo

go2_ctrl:
Domain ID = 1
Interface = lo
```

然后先使用 `dds_test` 验证 `rt/lowstate`。

---

# Next

安装完成后：

- 训练流程请阅读 README 中的 Training
- Sim-to-Sim 原理与完整步骤请阅读 `doc/Sim2Sim/`