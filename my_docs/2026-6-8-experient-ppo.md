# SFT 与 PPO 对权重的不同要求

## 模型结构差异

| 配置项 | SFT | PPO |
|--------|-----|-----|
| `add_value_head` | `False` | `True` |
| `train_expert_only` | `False`（全量微调） | 无此项，全量训练 |
| `noise_level` | 无 | 有（默认 0.3） |
| `detach_critic_input` | `True`（无效，无 value head） | `True` |

PPO 比 SFT 多一个 **value head**：一个小型 MLP（hidden sizes: 1024→512→256），用于估计当前状态的价值函数 V(s)，是 PPO advantage 计算的基础。SFT 不需要 value function，因此不挂载这个模块。

---

## 权重文件差异

### SFT checkpoint（`full_weights.pt`）

包含的参数：
- `paligemma_with_expert.*`：PaliGemma VLM + Gemma expert 全部参数
- 不含任何 `value_head.*` 键

保存路径结构：
```
checkpoints/global_step_XXXX/
└── actor/
    ├── model_state_dict/
    │   └── full_weights.pt      ← 可直接 torch.load 的完整权重
    └── dcp_checkpoint/          ← FSDP 分片格式，用于 resume 训练
        ├── __0_0.distcp
        ├── __1_0.distcp
        └── .metadata
```

### PPO checkpoint

包含的参数：
- `paligemma_with_expert.*`：同 SFT
- `value_head.*`：value head MLP 参数（PPO 独有）

保存路径结构（与 SFT 相同格式）：
```
checkpoints/global_step_XXXX/
└── actor/
    ├── model_state_dict/
    │   └── full_weights.pt
    └── dcp_checkpoint/
```

---

## 加载方式

### 用 pi05_base 从头初始化（SFT / PPO 均适用）

```yaml
actor:
  model:
    model_path: "/mnt/petrelfs/muyao/jiayusun/pi05_base"
runner:
  ckpt_path: null
  resume_dir: null
```

`model_path` 指向官方 pi05_base 目录，包含：
- HuggingFace 模型权重（`model.safetensors` 或 `pytorch_model.bin`）
- tokenizer 配置
- `physical-intelligence/robotwin/norm_stats.json`

### 用 SFT checkpoint 初始化 PPO（本项目当前设置）

```yaml
actor:
  model:
    model_path: "/mnt/petrelfs/muyao/jiayusun/pi05_base"   # 提供 tokenizer + norm_stats
    add_value_head: True                                     # PPO 需要
runner:
  ckpt_path: ".../global_step_15000/actor/model_state_dict/full_weights.pt"  # 覆盖主干权重
```

加载流程：
1. 按 `model_path` 初始化完整模型（含 value head，随机初始化）
2. 用 `ckpt_path` 的 `full_weights.pt` 覆盖权重（`strict=False`）
3. value head 因为在 SFT checkpoint 里不存在，**保持随机初始化**
4. 其余主干参数全部来自 SFT 训练结果

> **为什么用 `strict=False`**：SFT 的 `full_weights.pt` 没有 `value_head.*` 键，而 PPO 模型有。`strict=True`（默认）遇到 missing keys 会直接报错，改为 `strict=False` 后 missing keys 只打印日志，不影响加载。

### 用已有 PPO checkpoint 继续训练（resume）

```yaml
runner:
  resume_dir: ".../checkpoints/global_step_XXXX"   # 含 optimizer、scheduler 状态
  ckpt_path: null
```

`resume_dir` 会恢复完整训练状态（模型权重 + optimizer + lr_scheduler + data loader），用于中断后继续训练。

### 用 PPO checkpoint 初始化新 PPO（换任务或超参）

```yaml
actor:
  model:
    model_path: "/mnt/petrelfs/muyao/jiayusun/pi05_base"
runner:
  ckpt_path: ".../global_step_XXXX/actor/model_state_dict/full_weights.pt"
  resume_dir: null
```

PPO checkpoint 的 `full_weights.pt` 包含 value head，此时 `strict=True` 也能正常加载（所有键都匹配）。

---

## Value Head 随机初始化的影响

### 更新方式

PPO 的总 loss 分两部分，value head 通过 **value loss** 独立更新：

```
total_loss = actor_loss + value_loss
value_loss = MSE( V(s), Return )
```

每个 PPO update epoch 都会计算 value loss 并反向传播更新 value head 参数。value head 只是一个小型 MLP（1024→512→256→1），学习目标是预测标量回报，收敛速度远快于 actor。

### 随机初始化的问题

训练初期 V(s) 预测混乱 → advantage = Return - V(s) 估计不准 → actor 更新方向有噪声。

### 为什么不会崩溃

三个机制共同保护训练稳定性：

1. **PPO clipping**（`clip_ratio_high/low: 0.2`）
   无论 advantage 估计多离谱，策略单步更新幅度被限制在 ±20%，防止策略突变。

2. **GAE 平滑**（`gamma=0.99, gae_lambda=0.95`）
   Advantage 是多步折扣累积，不是直接用单步 V(s)，对 value head 早期误差有平均效果。

3. **Actor 权重来自 SFT，起点已经很好**
   策略本身已有完成任务的能力，PPO 只是微调提升成功率，对 advantage 精度要求没有从零训练那么高。

### 实践结论

随机初始化 value head 是所有 PPO 实现的标准做法，几十到几百个 step 后 value head 就会收敛到合理水平。无需特殊处理。

如果希望加速早期稳定性，可在 config 里设置：

```yaml
optim:
  critic_warmup_steps: 200  # 前 200 步只更新 value head，不更新 actor
```

### 当前实验配置

本项目 PPO config（`robotwin_adjust_bottle_ppo_openpi_pi05.yaml`）中：

```yaml
optim:
  critic_warmup_steps: 0  # 未启用，value head 与 actor 同步更新
```

Job **9742588** 使用 SFT step-15000 权重初始化，已提交（2026-06-08）。

---

## 总结

| 场景 | model_path | ckpt_path | resume_dir | strict |
|------|-----------|-----------|------------|--------|
| SFT 从头训练 | `pi05_base` | null | null | — |
| PPO 从头训练 | `pi05_base` | null | null | — |
| **SFT → PPO 初始化** | `pi05_base` | SFT `full_weights.pt` | null | `False`（value head missing） |
| PPO checkpoint → 新 PPO | `pi05_base` | PPO `full_weights.pt` | null | `True` |
| PPO 断点续训 | `pi05_base` | null | PPO checkpoint 目录 | — |
