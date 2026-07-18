# RoboTwin OpenPI JAX 与 RLinf PI05 SFT 训练差异

日期：2026-06-14

## 1. 对比对象

RoboTwin 官方训练：

```text
代码：/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05
入口：scripts/train.py
配置：pi05_aloha_full_base
后端：OpenPI JAX
```

RLinf 训练：

```text
代码：/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
入口：examples/sft/train_vla_sft.py
配置：examples/sft/config/robotwin_sft_openpi_pi05.yaml
OpenPI 配置：pi05_aloha_robotwin
后端：PyTorch + FSDP
```

两边都使用数据集：

```text
/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_47_pi05
```

## 2. 差异汇总

| 项目 | RoboTwin JAX | RLinf PyTorch | 影响判断 |
|---|---|---|---|
| 数据集 | `robotwin_47_pi05` | 相同 | 一致 |
| 训练步数 | 20k | 30k | 中 |
| global batch size | 64 | 64 | 一致 |
| 模型配置 | `pi05_aloha_full_base` | `pi05_aloha_robotwin` | 高 |
| `adapt_to_pi` | `False` | `True` | 很高 |
| joint delta action | 开启 | 开启 | 一致 |
| EMA | `ema_decay=0.99` | 无 | 很高 |
| 可训练参数精度 | FP32 参数，BF16 计算 | 大部分 Transformer 参数为 BF16 | 很高 |
| 图像增强 | JAX Augmax | PyTorch 手写实现 | 中高 |
| 优化器 | AdamW | AdamW | 基本一致 |
| flow matching loss | MSE | MSE | 基本一致 |
| 初始化权重 | JAX `pi05_base` | 转换后的 PyTorch `pi05_base` | 待严格验证 |

## 3. `adapt_to_pi` 表示空间不同

RoboTwin 官方配置：

```python
LeRobotAlohaDataConfig(
    repo_id="robotwin_47_pi05",
    adapt_to_pi=False,
)
```

RLinf 的 `pi05_aloha_robotwin` 没有覆盖 `adapt_to_pi`，而
`LeRobotAlohaDataConfig` 的默认值为：

```python
adapt_to_pi: bool = True
```

RLinf 因此会额外执行：

1. 翻转部分关节维度的符号。
2. 将 Aloha 夹爪数值转换到 PI 内部角度空间。
3. 在转换后的 state/action 空间中计算并应用 norm stats。
4. 使用转换后的 state 生成 PI0.5 离散 state token。

关节符号翻转 mask 为：

```text
[1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]
```

两份 norm stats 已确认处于不同表示空间。例如：

```text
state mean 第 2 维：
RoboTwin JAX:  +1.1588
RLinf:         -1.1588

state mean 左夹爪：
RoboTwin JAX: 0.6592
RLinf:        0.2481
```

这不是普通浮点误差，而是训练目标和状态表示发生了系统性变化。

## 4. EMA 差异

OpenPI JAX 默认配置：

```python
ema_decay = 0.99
```

训练过程中每一步都会更新 EMA：

```text
ema = 0.99 * old_ema + 0.01 * current_params
```

JAX checkpoint 保存给推理使用的 `params` 时，会优先保存 EMA 参数。因此
RoboTwin JAX 评测使用的是 EMA 权重。

RLinf SFT 当前：

```text
不维护 EMA
checkpoint 保存当前即时模型参数
评测直接加载即时参数
```

所以即使两边训练数据、步数和 optimizer 完全一致，最终评测的参数仍然不
等价。EMA 对 flow/diffusion 模型的推理稳定性可能有明显影响。

## 5. 可训练参数精度差异

RoboTwin JAX：

```text
模型计算使用 bfloat16
可训练参数保持 float32
Adam optimizer state 保持高精度
```

只有被冻结参数才会在初始化时转换为 BF16。`pi05_aloha_full_base` 是 full
finetune，没有冻结模型参数。

RLinf 在创建 optimizer 前调用：

```python
model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
```

这会将大部分 PaliGemma 和 action expert 参数直接转成 BF16。当前配置又是：

```yaml
train_expert_only: false
```

因此 RLinf 会直接更新大量 BF16 权重。可能产生以下影响：

1. 小梯度更新被 BF16 舍入。
2. full finetune 时 VLM 参数更新精度降低。
3. optimizer state 和参数更新的数值行为与 JAX FP32 master weights 不同。

这是当前解释训练质量差异的高优先级因素。

## 6. 图像增强差异

两边训练时都包含：

```text
非 wrist 相机随机裁剪到原图 95%
resize 回 224x224
随机旋转约 ±5°
brightness / contrast / saturation augmentation
```

但具体实现不同：

| 项目 | JAX | PyTorch |
|---|---|---|
| 增强库 | Augmax | 手写 Torch 操作 |
| resize | JAX image tools | `torch.interpolate` |
| rotation | Augmax Rotate | `grid_sample` |
| 边界处理 | Augmax 语义 | zero padding |
| 颜色增强 | Augmax | 手写缩放公式 |
| 随机参数 | JAX RNG | Torch RNG，部分参数整批共享 |

因此增强类型接近，但训练图像分布并非严格一致。

## 7. 基本一致的训练设置

以下项目已经基本对齐：

```text
数据集：robotwin_47_pi05
相机：head + left wrist + right wrist
action horizon：50
action dim：14
global batch size：64
joint action：转换为相对当前 state 的 delta
gripper action：保持 absolute
AdamW beta1：0.9
AdamW beta2：0.95
Adam epsilon：1e-8
weight decay：1e-10
peak learning rate：2.5e-5
warmup steps：1000
minimum learning rate：2.5e-6
gradient clipping：1.0
loss：flow matching MSE
训练范围：full finetune
```

JAX loss 先对 action dimension 求均值，再对 batch 和 action horizon 求均值；
PyTorch 对全部 batch、horizon 和 action dimension 直接求均值，数学上等价。

## 8. 当前判断

RLinf 当前训练不是 RoboTwin 官方训练的等价 PyTorch 实现。

最可能影响成功率的差异按优先级排序：

1. **JAX 使用 EMA 权重，RLinf 不使用 EMA。**
2. **JAX 保留 FP32 可训练参数，RLinf 直接更新大量 BF16 参数。**
3. **RoboTwin 使用 `adapt_to_pi=False`，RLinf 使用
   `adapt_to_pi=True`。**
4. JAX 与 PyTorch 的训练图像增强实现不完全相同。
5. JAX 训练 20k steps，RLinf 训练 30k steps。
6. JAX base checkpoint 转换为 PyTorch 后是否逐层等价尚未完全验证。

这些训练侧差异比 RLinf/native 测评框架差异更有能力解释当前结果：

```text
RoboTwin JAX SFT 20k：约 51%
RLinf PyTorch SFT：约 29%～36%
RoboTwin native + PyTorch SFT：约 26%
```

## 9. 建议的对齐实验

新建一组严格对齐的 PyTorch SFT：

```text
adapt_to_pi=False
使用对应 adapt_to_pi=False 的 norm stats
FP32 master parameters
ema_decay=0.99
global batch size=64
训练 20k steps
保存并评测 EMA checkpoint
```

实验顺序建议每次只改变一个变量：

1. 现有训练配置增加 EMA，评测 20k EMA 权重。
2. 保留 EMA，将可训练参数改为 FP32。
3. 使用 `adapt_to_pi=False` 及对应 norm stats。
4. 最后对齐 JAX 与 PyTorch 图像增强。

每组实验使用同一批 96 个 `seed + instruction` 严格配对评测，避免将环境
随机性误判为训练改进。

## 10. 对齐实验执行记录

### 10.1 能力检查

- RLinf OpenPI SFT 当前没有实际接入 EMA；配置中的 `ema_decay` 不会被
  SFT runner 使用，也不会保存 EMA checkpoint。
- 当前优化器直接更新模型参数，没有 FP32 master parameters；OpenPI
  加载阶段会将大部分可训练参数转为 BF16。
- 本轮不修改训练代码，暂时跳过 EMA 和 FP32，只执行框架现有能力支持的
  `adapt_to_pi=False` 对齐实验。

### 10.2 实验配置

```text
Config:
examples/sft/config/robotwin_sft_openpi_pi05_adapt_false.yaml

Slurm:
examples/sft/slurm/robotwin_pi05_sft_adapt_false_8gpu.sbatch

基础权重:
/mnt/petrelfs/muyao/jiayusun/pi05_base

实验资产目录:
/mnt/petrelfs/muyao/jiayusun/pi05_base_adapt_false

官方 adapt_to_pi=False norm stats:
/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/assets/
pi05_aloha_full_base/robotwin_47_pi05/norm_stats.json
```

实验参数：

```text
adapt_to_pi=False
global batch size=64
micro batch size=32
max steps=20000
save interval=5000
seed=0
EMA=不支持，未启用
FP32 master parameters=不支持，未启用
```

独立资产目录通过符号链接复用 `pi05_base` 权重，并将
`physical-intelligence/robotwin/norm_stats.json` 指向官方
`adapt_to_pi=False` stats，不修改原有 `pi05_base`。

### 10.3 提交状态

| 日期 | Job | 状态 | 说明 |
|---|---:|---|---|
| 2026-06-14 | 9788009 | RUNNING | 8 GPU，47 tasks，20k steps |

运行前检查：

- sbatch shell 语法检查通过。
- Hydra 配置合并通过，确认 `adapt_to_pi=false`、batch 64、20k steps。
- 实验 stats 与 RoboTwin 官方 stats 的 SHA256 均为
  `cd98666e0d2fa3d6d5ec28cad49c4186fce2beafa0b1916b91ec9d930478848a`。
- 登录节点禁用 Apptainer user namespace，完整 OpenPI 导入检查需在计算
  节点启动后由训练日志确认。

## 11. RoboTwin JAX 对齐到 RLinf 数据表示

### 11.1 实验目的

在保留 RoboTwin JAX 训练机制的情况下，只将 Aloha 数据表示改为 RLinf
默认方式：

```text
RoboTwin JAX 原配置：adapt_to_pi=False
本实验：adapt_to_pi=True
```

`adapt_to_pi=True` 会在统计和归一化前：

1. 翻转 state/action 的第 1、2、8、9 维关节符号。
2. 将左右 gripper（第 6、13 维）转换到 PI 预训练使用的角度表示。
3. 对机械臂关节 action 计算相对当前 state 的 delta；gripper 保持
   absolute。

norm stats 的统计算法没有改变，变化来自统计前的 state/action 坐标表示。

### 11.2 数据集

最终使用：

```text
/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/
robotwin_15_pi05
```

数据集元信息：

```text
tasks=15
episodes=1500
frames=359615
episodes per task=100
fps=50
action dim=14
```

曾错误提交 47-task 数据集，发现后立即取消：

| Job | 状态 | 说明 |
|---:|---|---|
| 9788130 | CANCELLED | 错误使用 `robotwin_47_pi05`，仅运行 10 分 24 秒 |
| 9788132 | CANCELLED | 9788130 对应的自动 PyTorch 转换任务 |

### 11.3 JAX 训练配置

RoboTwin 仓库：

```text
/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05
```

新增配置：

```text
config name=pi05_aloha_robotwin15_adapt_pi
repo_id=robotwin_15_pi05
adapt_to_pi=True
global batch size=64
num train steps=20000
num workers=8
save interval=5000
seed=42
ema_decay=0.99
full finetune
```

除 `adapt_to_pi`、数据集和独立配置名外，训练机制保持 RoboTwin JAX 原生
实现，包括 FP32 可训练参数、BF16 计算、EMA 和 JAX 图像增强。

norm stats 将写入：

```text
RoboTwin/policy/pi05/assets/
pi05_aloha_robotwin15_adapt_pi/robotwin_15_pi05/norm_stats.json
```

JAX checkpoint 将写入：

```text
RoboTwin/policy/pi05/checkpoints/
pi05_aloha_robotwin15_adapt_pi/robotwin_15_pi05_adapt_pi_full/
```

### 11.4 JAX 转 PyTorch

转换使用 OpenPI 官方脚本：

```text
/mnt/petrelfs/muyao/jiayusun/openpi/examples/
convert_jax_model_to_pytorch.py
```

Slurm 脚本：

```text
RoboTwin/policy/pi05/convert_robotwin15_adapt_pi_to_pytorch.sbatch
```

JAX checkpoint 的 `params` 由 RoboTwin checkpoint 逻辑保存为 EMA 参数。
转换输出为 BF16 `model.safetensors`，并复制匹配的 norm stats assets：

```text
RoboTwin/policy/pi05/checkpoints/
pi05_aloha_robotwin15_adapt_pi/
robotwin_15_pi05_adapt_pi_full_pytorch/20000
```

### 11.5 当前任务

| 日期 | Job | 状态 | 说明 |
|---|---:|---|---|
| 2026-06-14 | 9788134 | RUNNING | JAX，15 tasks，adapt_to_pi=True，20k |
| 2026-06-14 | 9788135 | PENDING (Dependency) | 9788134 成功后自动转 PyTorch |

注意：Job 9788009 使用 47-task 数据集，Job 9788134 使用 15-task 数据集。
两者不能作为严格训练框架对照。后续若要比较 JAX 与 RLinf PyTorch，必须
让两边都使用 `robotwin_15_pi05`、相同训练步数和同一批配对评测样本。
