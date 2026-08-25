
这一阶段的目标是：**将 IsaacLab + RSL-RL 中训练得到的 PPO checkpoint，转换成可以脱离训练环境独立推理的部署策略。**

本项目训练任务为：

```
Unitree-Go2-Adaptive-Energy-Flat
```

最终使用：

```
model_9999.pt
```

`model_9999.pt` 是 RSL-RL 的训练 checkpoint，其中除了 Actor，还可能包含 Critic、Optimizer 和训练状态等信息，主要用于继续训练，并不适合直接交给 C++ `go2_ctrl` 部署。因此使用 `play.py` 加载 checkpoint，并导出推理模型：

```
model_9999.pt
      ↓
play.py 加载 checkpoint
      ↓
恢复 Actor-Critic
      ↓
提取部署所需 Actor
      ↓
┌───────────────┬────────────────────┐
↓               ↓
policy.pt       policy.onnx
TorchScript     ONNX
                     +
              policy.onnx.data
```

最终生成：

```
exported/
├── policy.pt
├── policy.onnx
└── policy.onnx.data
```

其中：

- `policy.pt`：TorchScript 格式的 Actor，适合 PyTorch/LibTorch 推理。
- `policy.onnx`：ONNX 网络计算图，当前 `go2_ctrl` 使用该文件进行推理。
- `policy.onnx.data`：ONNX 的外部权重文件，必须与 `policy.onnx` 一起保留。

同时还需要：

```
params/deploy.yaml
```

它描述 Observation、历史长度、Action Scale、关节映射、默认关节位置、PD 参数和控制周期等部署接口。

## 1.1 Actor 网络结构

导出的 ONNX 模型经过检查：

```
Input:  obs [1, 1350]

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
Output: actions [1, 12]
```

其中：

```
1350 = 45维单帧观测 × 30帧历史
12   = Go2 的 12 个关节动作
```

网络共有：

```
857,484 parameters
```

因此阶段一最终完成的是：

```
IsaacLab / RSL-RL Training
          ↓
    model_9999.pt
          ↓
      Actor Export
          ↓
policy.onnx + policy.onnx.data
          ↓
1350维 Observation
          ↓
       Actor
          ↓
12维 Action
```

需要注意，**ONNX 输出的 12 维 Action 还不是最终电机目标角度**。后续部署阶段还要结合 `deploy.yaml` 中的 `action_scale`、`default_joint_pos` 等参数生成真正的 `q_des`。