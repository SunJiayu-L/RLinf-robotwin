# RLinf / RoboTwin Eval 差距排查计划

日期：2026-06-11

## 1. 背景

`adjust_bottle` 当前观测结果：

| 评测框架 | 权重 | 成功率 |
|---|---|---:|
| RLinf eval | PyTorch SFT 30k | 21.9%（21/96） |
| RoboTwin native `eval_policy.py` | PyTorch SFT 30k | 37.0%（37/100） |
| RoboTwin native eval | JAX SFT 20k | 54.5%（两轮均值） |

上述结果不是严格控制变量实验：

- RLinf 使用 32 个固定 seed，重复 3 轮，共 96 条轨迹。
- native PyTorch 使用 100 个不同且通过 expert 检查的 episode。
- 两边的 instruction 类型、planner backend、最大步数和运行方式没有完全对齐。
- PyTorch 30k 与 JAX 20k 不是同一 checkpoint，也不是简单格式转换关系。

因此当前不能将 15.1 pp 差距直接归因于 RLinf 框架，也不能将 17.5 pp
差距直接归因于 PyTorch/JAX 格式差异。

## 2. 已排除或降级的假设

### 2.1 `num_steps=50` 导致零动作

RLinf 在 `num_steps=50` 和 `num_steps=10` 下的成功率分别为 22.9% 和
21.9%，没有支持该假设。

原日志中的 `action_arm_range` 只统计左臂动作，不能据此判断整个双臂系统
是否产生零动作。

### 2.2 RLinf 环境执行了额外裁剪或图像增强

当前 eval 配置中 `center_crop: false`。RLinf 环境仅额外执行 PIL
`convert("RGB")`，输入为 RGB `uint8` 时通常不会改变像素。

RLinf PyTorch native wrapper 和 RLinf eval 都调用同一个
`predict_action_batch()`，因此 21.9% 与 37.0% 的差距不应归因于模型内部
图像预处理。

PyTorch 与 JAX 的 resize 实现不同，可能造成小幅数值差异，但需要单独验证，
不能作为当前结论。

## 3. 总体排查策略

排查分成两个阶段：

1. 使用同一份 PyTorch SFT 30k 权重，对齐 RLinf 和 RoboTwin native 的所有
   评测条件，定位 21.9% 到 37.0% 的框架差距。
2. 在完全相同的评测协议下，比较 PyTorch 与 JAX checkpoint，定位 37.0%
   到 54.5% 的模型、训练配置和推理后端差距。

每次实验只改变一个变量，并记录逐 episode 配对结果。

## 4. 阶段一：RLinf 与 native PyTorch 严格对齐

### 4.1 对齐评测协议

两边统一以下条件：

| 条件 | 统一值 |
|---|---|
| checkpoint | PyTorch SFT step 30000 |
| episode seed | 同一份 100 seed 清单 |
| instruction | 每个 seed 使用完全相同的文本 |
| instruction type | `unseen` |
| episode 上限 | `step_lim=400` |
| planner backend | 两边使用同一个 backend |
| 相机配置 | D435，head + left wrist + right wrist |
| domain randomization | 完全相同且显式记录 |
| action chunk | 完全相同 |
| denoise steps | `num_steps=10` |

不要使用 RLinf 当前“32 个 seed 重复 3 次”的统计作为配对实验结果。

### 4.2 增加逐 episode 日志

RLinf 和 native 都记录：

```text
episode_id
seed
instruction
planner_backend
success
success_step
termination_reason
total_steps
```

输出 CSV 或 JSONL，便于按 seed 直接 join。重点统计：

- 两边都成功
- 两边都失败
- native 成功、RLinf 失败
- RLinf 成功、native 失败

### 4.3 比较同一 episode 的首次模型输入

选择至少 10 个配对 seed，保存：

- 环境原始 head、left wrist、right wrist RGB
- 进入 `predict_action_batch()` 前的 tensor
- 模型 resize/pad 后的 `224x224` tensor
- 原始 state 和标准化后的 state
- instruction 和 token IDs

图像比较指标：

```text
shape
dtype
min / max
channel order
max_abs_diff
mean_abs_diff
MSE
```

判定标准：

- PIL 往返前后应逐像素一致。
- 两条 PyTorch 路径进入模型前的图像和 state 应完全一致。
- 若不一致，从第一个出现差异的转换步骤继续定位。

### 4.4 固定扩散噪声比较 action chunk

对相同 observation：

1. 固定 PyTorch、NumPy 和 CUDA 随机种子。
2. 最好直接构造并传入相同的初始 diffusion noise。
3. 比较两条路径输出的完整 action chunk。

记录：

```text
noise hash
normalized state
image tensor hash
token IDs
raw model actions
post-transform actions
```

如果输入、noise 和权重相同，action 应在浮点误差范围内一致。

### 4.5 比较环境执行轨迹

若模型 action 一致但成功结果不同，对同一 seed 逐 step 记录：

```text
requested action
executed action
left/right joint state
left/right gripper state
reward
success
termination
```

重点检查：

- action chunk 的执行长度和重新推理频率
- action 是否被裁剪、重排或类型转换
- `ignore_terminations` 与 auto-reset 行为
- success 判定时机
- 环境版本和任务代码差异
- Ray 并行环境是否引入 seed 或 reset 状态偏移

## 5. 阶段二：PyTorch 与 JAX 差距

阶段一完成后，固定同一套 100 seed 协议，再分别运行：

| 实验 | checkpoint/runtime | 目的 |
|---|---|---|
| A | PyTorch SFT 20k | 排除 checkpoint step 差异 |
| B | PyTorch SFT 30k | 测量训练步数影响 |
| C | JAX SFT 20k | 测量训练管线和后端综合差异 |

需要同时核对：

- `adapt_to_pi`
- state/action 编码与关节符号变换
- norm stats 文件及其统计范围
- 训练数据和 data config
- checkpoint 是否来自同一次训练
- PyTorch/JAX 权重是否真正对应
- bfloat16/float32 精度
- JAX 与 PyTorch resize/pad 的像素差异
- 相同 noise 下的中间网络输出和 action

只有使用同源 checkpoint 转换后，才能将差距称为“格式或运行时差异”。

## 6. 推荐执行顺序

1. 生成唯一的 100 seed + instruction 清单。
2. 修改两边日志，输出逐 episode JSONL。
3. 统一环境配置，重跑 native PyTorch 30k。
4. 使用相同清单重跑 RLinf PyTorch 30k。
5. 对两份 JSONL 做配对统计。
6. 对不一致 seed 保存首次输入和 action chunk。
7. 固定 noise，验证策略输出是否一致。
8. 若输出一致，比较逐 step 环境轨迹。
9. 阶段一闭环后，再运行 PyTorch 20k 与 JAX 20k 对照。

## 7. 根因判定标准

| 观察结果 | 根因方向 |
|---|---|
| 首次模型输入不同 | observation、图像、state 或 instruction 管线 |
| 输入相同但固定 noise 后 action 不同 | 权重加载、模型模式、精度或推理实现 |
| action chunk 相同但执行轨迹不同 | 环境 wrapper、action 执行或任务代码 |
| 配对实验结果接近 | 原差距主要来自 seed 和评测协议 |
| 同源 PT/JAX 权重仍有稳定差距 | resize、数值精度或后端实现 |
| PT20k 接近 JAX20k，PT30k 较差 | checkpoint 或训练过程退化 |

## 8. 最终交付物

- 固定的 100 episode seed/instruction 清单
- RLinf 与 native 的逐 episode JSONL
- 配对成功率和 disagreement seed 列表
- 同一 observation 的图像/state/token/action 对比报告
- 典型 disagreement episode 的逐 step 轨迹
- 最终根因、影响比例及修复后的回归结果
