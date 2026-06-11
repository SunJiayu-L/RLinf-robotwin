# PI05 RoboTwin PPO 实验记录

**日期**：2026-06-07 ～ 2026-06-12
**模型**：PI05（`pi05_base`）
**平台**：RLinf + RoboTwin-RLinf
**机械臂**：aloha-agilex

---

## 1. 准备工作

### 1.1 Norm Stats 计算

PI05 官方 norm_stats 由 `adjust_to_pi=True` 的变换计算，与旧 openpi 不兼容，需重新计算。

- **数据集**：`robotwin_47_pi05`（路径：`/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_47_pi05`）
- **计算 Job**：9729926，COMPLETED（2026-06-06 22:29 ～ 2026-06-07 16:14）
- **输出路径**：`/mnt/petrelfs/muyao/jiayusun/pi05_base/physical-intelligence/robotwin/norm_stats.json`
- **状态**：✅ 文件现存，包含 `state` 和 `actions` 两个字段

---

## 2. SFT 训练

### 2.1 配置文件

- **Config**：`examples/sft/config/robotwin_sft_openpi_pi05.yaml`
- **模型路径**：`/mnt/petrelfs/muyao/jiayusun/pi05_base`
- **数据**：`robotwin_47_pi05`
- **总步数**：30,000 steps（`save_interval: 5000`）
- **硬件**：8 GPU，`global_batch_size: 64`，`micro_batch_size: 32`
- **学习率**：2.5e-5，cosine decay，warmup 1000 steps

### 2.2 训练历史

| Run | Job ID | 状态 | 时长 | 备注 |
|-----|--------|------|------|------|
| 第1次（4-GPU） | 9737302 | FAILED (OOM) | ~6.3h | step 5000 保存后 backward OOM |
| 第1次（8-GPU） | 9737305 | TIMEOUT | 24h | 跑到 ~step 21,358，time limit 到期 |
| Resume（8-GPU） | 9743981 | COMPLETED ✅ | ~11h | 从 step 20,000 续跑到 30,000 |

### 2.3 Resume 要点

Hydra struct 问题：`resume_dir` 未在 YAML struct 中定义，需要用 `+` 前缀追加：
```bash
+runner.resume_dir="${RESUME_DIR}"   # 正确
runner.resume_dir="${RESUME_DIR}"    # 错误：Key 'resume_dir' is not in struct
```

### 2.4 权重丢失事件（2026-06-09）

SFT 完成后，`logs/20260607-17:10:39-robotwin_sft_pi05_8gpu/` 整个目录消失，
所有 checkpoint（step 5000/10000/20000/25000/30000）全部丢失，手动误删。

**应对**：重新提交 SFT（Job 9747041），时限调整为 **48h**，可一次性完成 30,000 步。

### 2.5 当前 SFT Job

| Job ID | 状态 | 提交时间 | 时限 | sbatch |
|--------|------|----------|------|--------|
| 9747041 | RUNNING | 2026-06-09 | 48h | `examples/sft/slurm/robotwin_pi05_sft_8gpu.sbatch` |

---

## 3. PPO 配置

### 3.1 代码修改

**问题**：SFT checkpoint 不含 `value_head.*` 键，PPO 模型有，`strict=True` 加载报错。

**修改**（两处）：
- `rlinf/workers/actor/fsdp_actor_worker.py`：`load_state_dict(..., strict=False)`
- `rlinf/workers/rollout/hf/huggingface_worker.py`：`load_state_dict(..., strict=False)`

missing keys（value head）打印 INFO 日志，保持随机初始化，正常 PPO 做法。

### 3.2 任务配置一览

#### adjust_bottle（原始任务）

| 项目 | 值 |
|------|----|
| Config | `examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml` |
| sbatch | `examples/embodiment/slurm/robotwin_pi05_ppo.sbatch` |
| experiment_name | `robotwin_ppo_openpi_pi05` |
| 初始权重 | SFT `global_step_20000/actor/model_state_dict/full_weights.pt` |
| critic_warmup_steps | 0 |
| total_num_envs (train/eval) | 64 / 32 |
| 时限 | 336h（14天） |

#### beat_block_hammer（新增）

| 项目 | 值 |
|------|----|
| Config | `examples/embodiment/config/robotwin_beat_block_hammer_ppo_openpi_pi05.yaml` |
| sbatch | `examples/embodiment/slurm/robotwin_pi05_ppo_beat_block_hammer.sbatch` |
| experiment_name | `robotwin_ppo_openpi_pi05_beat_block_hammer` |
| 初始权重 | SFT `global_step_20000/actor/model_state_dict/full_weights.pt` |
| critic_warmup_steps | 0 |
| total_num_envs (train/eval) | 64 / 32 |
| 时限 | 336h（14天） |
| env config | `examples/embodiment/config/env/robotwin_beat_block_hammer.yaml`（已有） |

#### put_bottles_dustbin（新增）

| 项目 | 值 |
|------|----|
| Config | `examples/embodiment/config/robotwin_put_bottles_dustbin_ppo_openpi_pi05.yaml` |
| sbatch | `examples/embodiment/slurm/robotwin_pi05_ppo_put_bottles_dustbin.sbatch` |
| experiment_name | `robotwin_ppo_openpi_pi05_put_bottles_dustbin` |
| 初始权重 | SFT `global_step_20000/actor/model_state_dict/full_weights.pt` |
| critic_warmup_steps | 0 |
| total_num_envs (train/eval) | 64 / 32 |
| 时限 | 336h（14天） |
| env config | `examples/embodiment/config/env/robotwin_put_bottles_dustbin.yaml`（**新建**） |

新建 env config 路径：`examples/embodiment/config/env/robotwin_put_bottles_dustbin.yaml`，
对应任务 Python 实现：`RoboTwin-RLinf/envs/put_bottles_dustbin.py`（已存在）。

### 3.3 PPO 公共配置

所有三个任务的 PPO 共享以下配置：

```yaml
actor:
  model:
    model_path: "/mnt/petrelfs/muyao/jiayusun/pi05_base"
    num_action_chunks: 50
    action_dim: 14
    add_value_head: True
    openpi:
      config_name: "pi05_aloha_robotwin"
      num_images_in_input: 3
      noise_level: 0.3
      detach_critic_input: True
  optim:
    lr: 5.0e-06
    value_lr: 1.0e-04
    critic_warmup_steps: 0
algorithm:
  rollout_epoch: 4
  update_epoch: 5
  global_batch_size: 2048
env:
  task_config:
    embodiment: [aloha-agilex]   # 覆盖 env yaml 的默认 piper
    camera:
      collect_wrist_camera: true
    domain_randomization:
      random_background: false   # 关闭随机化，降低复杂度
```

---

## 4. PPO 遇到的问题

### 4.1 CPU OOM（256 envs）

**现象**：Job 9745479/9745668 跑 ~5h 后 Ray OOM killer 终止，`DefaultCPUAllocator: can't allocate memory`

**原因**：`total_num_envs: 256` 并发 RoboTwin 仿真实例占用大量 CPU 内存（每个实例需要仿真状态 + 图像缓冲区）。

**修复**：`total_num_envs: 256 → 64`（train），`128 → 32`（eval），降低 4 倍内存占用。

### 4.2 step-15000 权重不存在

**现象**：Job 9747000 启动 5 分钟后 `FileNotFoundError`，`global_step_15000/full_weights.pt` 不存在。

**原因**：8-GPU SFT 的 `save_interval: 5000`，保存的是 5000/10000/20000/25000/30000，没有 15000。

**修复**：三个 PPO config 统一改为 `global_step_20000`。

---

## 5. 当前状态（2026-06-11 更新）

### 5.1 SFT

| Job ID | 状态 | 进度 | 说明 |
|--------|------|------|------|
| 9747041 | **COMPLETED** ✅ | step 30000/30000 | 路径：`logs/20260609-10:17:50-robotwin_sft_pi05_8gpu/`，checkpoint 已保存 |

### 5.2 PPO 任务汇总

| Job ID | 任务 | 状态 | 已运行时长 | 备注 |
|--------|------|------|------|------|
| 9760196 | adjust_bottle | **RUNNING** | ~1d5h | 节点 54-56，step ~200+，成功率 62–67% |
| 9762126 | beat_block_hammer | **RUNNING** | ~1d1h | 节点 54-20，step ~170+，成功率 19–23% |
| 9763675 | place_empty_cup | **RUNNING** | ~17h | 节点 54-5，step ~150+，成功率 62–81% |
| 9763150 | place_container_plate | **RUNNING** | ~22h | 节点 54-51，step ~150+，成功率 50–61% |
| 9760202 | put_bottles_dustbin | **已取消** | — | RLinf eval 0%，无 reward 信号 |

### 5.3 place_empty_cup 提交历史（多次失败）

| Job ID | 结果 | 失败原因 |
|--------|------|----------|
| 9762516 | ❌ OOM crash | 节点 `SH-IDCA1404-10-140-54-118` OOM kill → DistNetworkError |
| 9762952 | ❌ OOM crash | 节点 `SH-IDCA1404-10-140-54-18` OOM kill → DistNetworkError |
| **9763675** | ✅ RUNNING | 已排除 `-118` 和 `-18` 两个坏节点 |

### 5.4 place_container_plate 提交历史

| Job ID | 结果 | 失败原因 |
|--------|------|----------|
| 9762517 | ❌ AssertionError | `global_batch_size: 1024`，rollout_size=96 不能被 batch_size_per_rank=128 整除 |
| **9763150** | ✅ RUNNING | `global_batch_size` 修复为 `768`（96÷96=1 ✅） |

---

## 6. SFT Baseline Eval 结果

### 6.1 第一轮 Eval（num_steps=50）

使用 SFT step-30000 权重，RLinf eval 框架，32 eval envs × 3 epochs = 96 trajectories：

| Task | Job ID | success_once | RoboTwin eval（参考） |
|---|---|---|---|
| adjust_bottle | 9777092 | **22.9%** (22/96) | 54.5% |

> 结果显著低于 RoboTwin eval，见 §9 根因分析。

### 6.2 第二轮 Eval（num_steps=10）—— 结果

全部 **COMPLETED**，结果如下：

| Job ID | 任务 | success_once | RoboTwin native eval（参考） | 差距 |
|--------|------|------|------|------|
| 9777186 | adjust_bottle | **21.9%** (21/96) | 54.5% | -33% |
| 9777187 | beat_block_hammer | **5.2%** (5/96) | 73.5% | -68% |
| 9777188 | put_bottles_dustbin | **0%** (0/96) | 47.0% | -47% |
| 9777189 | place_empty_cup | **30.2%** (29/96) | 79.5% | -49% |
| 9777190 | place_container_plate | **29.2%** (28/96) | 78.0% | -49% |

> **结论**：num_steps 从 50 改成 10 **没有改善成功率**，所有任务 RLinf eval 仍远低于 RoboTwin native eval。
> 此前根据 `action_arm_range` 统计的“近零动作比例”无效：该字段只统计左臂 6 维动作，不能代表双臂均为零。
> 见 §9 更新后的根因分析。

### 6.3 第三轮 Eval（RoboTwin native eval 框架 + RLinf PyTorch 权重）

**目的**：排查是 RLinf eval 框架有 bug，还是 SFT 权重本身有问题。

**修复过程**：
- Job 9777277：FAILED（`ffmpeg` not found，`eval_video_log: true`）
- Job 9777294：FAILED（`ffmpeg` not found，override `eval_video_log False` 无效，因为该字段来自 task config YAML 而非 deploy_policy.yml）
- **根因**：`eval_video_log` 从 `task_config/demo_clean.yml` 加载，override 只更新 `usr_args`，不影响 task config。
- **修复**：新建 `task_config/demo_clean_no_video.yml`（`eval_video_log: false`），sbatch 默认使用该 config。

**最终结果**（Job 9777330，adjust_bottle，SFT step-30000，100 episodes）：

| 框架 | 权重 | 成功率 |
|------|------|--------|
| RoboTwin native eval_policy.py | SFT 30k PyTorch | **37.0%** (37/100) |
| RLinf eval | SFT 30k PyTorch | **21.9%** (21/96) |
| RoboTwin native eval（参考）| SFT 20k JAX | 54.5% |

- 使用 `policy/pi05_rlinf/deploy_policy.py`（新建），与 RLinf 框架使用完全相同的 `predict_action_batch()` 逻辑
- **观察到的差距**：
  - RLinf eval → RoboTwin native (PyTorch)：+15.1 pp
  - RoboTwin native (PyTorch SFT 30k) → RoboTwin native (JAX SFT 20k 参考)：+17.5 pp
- 两段比较都同时改变了 seed、instruction、episode 协议或 SFT pipeline，
  目前不能分别归因于“eval 框架”和“权重格式”
- 详见 §9 更新后的根因分析

### 6.4 SFT 20k Baseline Eval（COMPLETED）

**目的**：PPO 从 SFT 20k 初始化，获取 SFT 20k 的准确 baseline 作为 merged 权重的公平对比基准。

| Job ID | 任务 | success_at_end |
|--------|------|----------------|
| 9777843 | adjust_bottle | **26.0%** (25/96) |
| 9777844 | place_container_plate | **27.1%** (26/96) |
| 9777845 | place_empty_cup | **28.1%** (27/96) |

---

## 6.5 评测方法说明

每个 task 测评 **96 次**（`total_num_envs: 32` × `eval_rollout_epoch: 3`）。
使用 `eval_seeds.json` 中预筛选的成功 seed（adjust_bottle: 150个，place_container_plate/place_empty_cup: 260个），
`use_fixed_reset_state_ids: True`，每轮 32 个不同 seed 并行，共 3 轮。

---

## 7. 任务选择策略

### 7.1 PPO 可行性判断

**核心原则**：PPO 需要非零 reward 信号，SFT baseline 成功率 < 5% 的任务 PPO 基本无效。

参考 `pi05_robotwin47_experiment_log.md`（JAX SFT 20k，100 episodes/task，demo_clean，seed=0，pi0_step=50）：

- 成功率 ≥ 10%：38 个任务，均可尝试 PPO
- 成功率 < 5%：8 个任务，不建议 PPO

### 7.2 已放弃任务

| 任务 | 原因 |
|---|---|
| **put_bottles_dustbin** | RLinf eval 0% success，PPO 无 reward 信号无法学习 |

### 7.3 当前运行任务

| 任务 | SFT baseline（RoboTwin） | 类型 |
|---|---|---|
| adjust_bottle | 54.5% | 单臂 |
| beat_block_hammer | 73.5% | 单臂 |
| place_empty_cup | 79.5% | 单臂 |
| place_container_plate | 78.0% | 单臂 |

### 7.4 后续候选任务（按优先级）

| 任务 | SFT baseline | 类型 | 状态 |
|---|---|---|---|
| shake_bottle_horizontally | **91.0%** | 单臂 | 待提交（需建 env config） |
| shake_bottle | **84.5%** | 单臂 | 待提交（需建 env config） |
| press_stapler | 76.5% | 单臂 | 待提交（需建 env config） |
| **stack_bowls_two** | **82.0%** | **双臂** | 待提交（双臂最高，需建 env config） |
| dump_bin_bigbin | 74.0% | 双臂 | 待提交（需建 env config） |

> 优先提交单臂高成功率任务；双臂任务从 stack_bowls_two 开始尝试。

---

## 8. 已知问题记录

| 问题 | 原因 | 状态 |
|---|---|---|
| CPU OOM（256 envs） | 并发仿真内存不足 | ✅ 已修复，改为 64/32 |
| step-15000 权重不存在 | save_interval=5000，无 15k | ✅ 已修复，统一用 20k |
| DistNetworkError（beat_block_hammer 9760201） | 节点 10.140.54.20 网络抖动 | ✅ 已重提为 9762126 |
| OOM crash on 10.140.54.118 | put_bottles_dustbin、place_empty_cup 均在此节点 OOM | ✅ 已加入 `--exclude` |
| OOM crash on 10.140.54.18 | place_empty_cup 9762952 在此节点 OOM | ✅ 已加入 `--exclude` |
| place_container_plate batch 不整除 | rollout_size=96，batch_size_per_rank=128，96%128≠0 | ✅ 已修复，global_batch_size 改为 768 |
| RLinf eval << RoboTwin eval（全任务约 20-30% vs 50-80%）| num_steps 未改善成功率；”零动作”统计无效；三段差距已分解：框架 ~15pp + 权重版本 ~17.5pp | 🔍 部分明朗；SFT 20k 在两框架下的对比待 Jobs 9777843-9777845 完成 |
| ffmpeg 缺失导致 RoboTwin native eval 崩溃 | 容器内无 ffmpeg，`eval_video_log: true` 触发 subprocess 调用 | ✅ 已修复：新建 `task_config/demo_clean_no_video.yml`，sbatch 默认使用 |

### 8.1 已知坏节点

| 节点 | 故障类型 | 影响任务 |
|------|----------|----------|
| `SH-IDCA1404-10-140-54-118` | GPU OOM kill（SIGKILL）| put_bottles_dustbin, place_empty_cup 9762516 |
| `SH-IDCA1404-10-140-54-18` | GPU OOM kill（SIGKILL）| place_empty_cup 9762952 |

所有新建 sbatch 均应加上：
```
#SBATCH --exclude=SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18
```

---

---

## 9. RLinf Eval << RoboTwin Eval 根因分析（更新版）

### 9.1 现象

所有 5 个任务 RLinf eval 结果均远低于 RoboTwin native eval：

| 任务 | RLinf eval (num_steps=50) | RLinf eval (num_steps=10) | RoboTwin native |
|---|---|---|---|
| adjust_bottle | 22.9% | **21.9%** | 54.5% |
| beat_block_hammer | — | **5.2%** | 73.5% |
| put_bottles_dustbin | — | **0%** | 47.0% |
| place_empty_cup | — | **30.2%** | 79.5% |
| place_container_plate | — | **29.2%** | 78.0% |

adjust_bottle 两轮 eval 结果基本一致，说明将 num_steps 从 50 改为 10
**不能解决成功率差距**。

### 9.2 已排除假设：num_steps 导致 ODE 模式坍塌

- 原假设：num_steps=50 步使 flow ODE 更精确地收敛到近零吸引子
- **已证伪**：num_steps=10 给出完全相同的结果（~21.9% vs ~22.9%）
- RoboTwin 配置中的 `pi0_step: 50` 是每次执行的 action chunk 长度，不是 ODE 去噪步数；OpenPI `sample_actions()` 未显式传参时仍使用默认 `num_steps=10`

### 9.3 无效诊断：将 `action_arm_range` 当作整机零动作

日志中的诊断实现为：

```python
chunk_actions[:, :6].min(), chunk_actions[:, :6].max()
```

- `chunk_actions[:, :6]` 仅包含左臂 6 个关节
- 左夹爪位于第 6 维，右臂位于第 7～12 维，右夹爪位于第 13 维
- RoboTwin 任务可能只使用右臂，或左右臂交替执行
- 因此左臂范围接近 0 不能说明机械臂不动，也不能据此排除 episode 后重新计算成功率
- 此前记录的“约 60% 零动作”应改称“约 60% episode 的首个 action chunk 左臂近零”，不具有失败归因意义

后续若要判断整机零动作，应同时记录左右臂：

```python
left_arm_range = chunk_actions[:, :6]
right_arm_range = chunk_actions[:, 7:13]
```

只有左右臂相对各自当前关节状态的变化都接近 0，才能判定为候选零动作 episode。

### 9.4 当前最可能的根因候选

**1. 图像预处理 dtype 问题**（最高优先）

推理路径中可能存在 dtype 不匹配：
- 环境输出 uint8 HWC 图像
- `input_transform` 中 `ResizeImages` 可能将 uint8 → float32
- 若图像在进入 `Observation.from_dict()` 时已是 float32，`dtype == torch.uint8` 判断不触发
- 图像以 [0,255] 或 [0,1] 范围进入模型（应为 [-1,1]）
- 需要直接打印模型输入 dtype/range 才能确认，目前没有证据证明该问题存在

**2. SFT checkpoint vs openpi JAX 基准**

- openpi JAX 参考值（54.5%）评估的是 SFT 20k 的 JAX 权重
- RLinf 评估的是 SFT 30k 的 PyTorch 权重
- 两者权重格式不同（JAX orbax vs PyTorch safetensors），加载过程有差异

**3. PPO 初期与 SFT eval 的一致性**

- PPO 训练 rollout 初期也只有约 20% 成功率
- PPO 经过 200 步训练后升至 62-67%，说明模型在 PPO 框架内能学习
- 这与 RLinf eval 22% 基线一致，PPO 就是从这个基线开始的

### 9.5 验证结果（2026-06-11 更新）

**Job 9777330**（RoboTwin native eval + RLinf SFT 30k PyTorch 权重，adjust_bottle，100 episodes）：

**结果：37.0%**（37/100）

客观结果：

```
RLinf eval (21.9%)
      ↕ +15.1 pp
RoboTwin native eval, PyTorch SFT 30k (37.0%)
      ↕ +17.5 pp
RoboTwin native eval, JAX SFT 20k 参考值 (54.5%)
```

这两个差值目前都**不能直接解释为单一因素的贡献**，因为比较中同时改变了多个变量。

#### 9.5.1 21.9% → 37.0%：评测协议不一致，不是纯框架 A/B

两次评测使用完全相同的 PyTorch SFT 30k 权重和
`predict_action_batch()`，说明模型在 RoboTwin native 路径中可以达到 37%，
RLinf 的 21.9% 不是单纯由 checkpoint 无法加载或模型完全失效造成。

但两套评测协议存在以下差异：

| 项目 | RLinf eval | RoboTwin native + PyTorch |
|---|---|---|
| episode 数量 | 32 个固定 seed × 3 次 rollout = 96 | 100 个不同的 expert-valid episode |
| seed 来源 | `eval_seeds.json`，seed 形如 `100100xxx` | 从 `100000` 开始动态 expert-check，实际 seed 形如 `100001` |
| seed 更新 | `use_fixed_reset_state_ids: true`，三轮重复相同 32 个场景 | 每个有效 episode 使用新的 seed |
| instruction 类型 | `VectorEnv` 默认 `seen` | `instruction_type: unseen` |
| planner backend | RLinf 配置显式使用 `mplib` | native 路径使用 Curobo/expert-check |
| episode 上限 | 200 environment steps | 400 environment steps |
| 执行方式 | 32 env Ray/VectorEnv 并行 | 单 env 串行 |

因此 15.1 pp 混合了 seed 难度、instruction、planner/setup、episode horizon
和并行环境差异，不能写成“RLinf/Ray 框架固定低估 15 pp”。

对 Job 9777330 的逐 episode 日志统计表明：

- 37 个成功中，35 个在 200 步以内完成
- 只有 2 个在 200 步以后完成（step 229、327）
- 因此 200 vs 400 的 episode 上限最多解释该次 native 结果中的约 2 pp，
  不是 15.1 pp 差距的主要来源

**当前最高优先根因：RLinf eval 不是 96 个独立、与 native 同分布的
episode，而是 32 个固定 seed 重复三次。** 在统一 seed 和 instruction
之前，无法判断剩余差距是否来自 VectorEnv/Ray。

#### 9.5.2 37.0% → 54.5%：不是单纯 JAX→PyTorch 格式损失

54.5% 是两次 JAX native eval（52/100、57/100）的平均值；PyTorch
30k 目前只有一次 37/100。两者除运行时外，还存在：

- checkpoint 不同：JAX SFT 20k vs RLinf PyTorch SFT 30k
- 训练过程不同：PyTorch 权重是独立 SFT 训练结果，不是 JAX 20k 权重的直接格式转换
- 数据变换不同：RLinf 使用 `pi05_aloha_robotwin`、`adapt_to_pi=True`
- JAX 原生训练使用 `pi05_aloha_full_base`、`adapt_to_pi=False`
- 两者使用各自重新计算的 norm stats，状态和动作表示并不相同
- JAX 与 PyTorch 模型实现、数值精度和随机噪声序列不同
- 参考 JAX eval 位于 `RoboTwin`，PyTorch native eval 位于
  `RoboTwin-RLinf`，环境代码版本也未锁定为完全一致

所以 17.5 pp 应描述为“两个不同 SFT pipeline 的结果差距”，不能归因为
checkpoint 格式转换损失。与平均值相比，使用同为 seed=0 的 JAX 52%
作为参考更合理，此时观察差距为 15 pp，但仍未隔离训练步数和数据变换。

#### 9.5.3 已降低优先级的假设

- **图像 dtype/归一化错误**：同一 PyTorch 模型和同一
  `predict_action_batch()` 在 native 路径达到 37%，说明通用模型预处理
  并未完全损坏。RLinf 与 wrapper 都从 HWC uint8 图像构造输入，该假设
  仍可通过打印 transform 前后 dtype/range 验证，但不再是最高优先级。
- **num_steps**：10 和 50 的 RLinf 结果基本一致，已排除为主要原因。
- **episode horizon**：根据 Job 9777330，只能解释约 2 pp。

#### 9.5.4 根因验证顺序

1. 固定同一批 100 个 seed 和对应 instruction，在 native 与 RLinf 中逐
   episode 对齐评测；两边统一 `step_lim=400`、`instruction_type=unseen`
   和 planner backend。
2. 记录每个 episode 的 `seed`、instruction、成功标记、成功 step，计算
   paired disagreement，而不是只比较总体成功率。
3. 在上述统一协议下比较同一 PyTorch 30k 权重，才能量化
   VectorEnv/Ray 的真实影响。
4. 再在完全相同的 native 协议下比较 PyTorch 20k、PyTorch 30k 和 JAX
   20k；只有这样才能依次隔离训练步数、实现和数据变换差异。
5. Jobs 9777843-9777845 完成后，优先加入 PyTorch 20k 的同协议结果。

---

---

## 10. 多任务权重融合实验（Task Arithmetic）

### 10.1 背景与动机

三个 PPO 任务（adjust_bottle、place_container_plate、place_empty_cup）均从 SFT step-20k 初始化，
各自训练到 step-150。Task Arithmetic 将三路 PPO task vector 叠加到 SFT 基底，
期望得到在三个任务上均有提升的通用模型，无需分别部署。

### 10.2 方法

**公式**：
```
θ* = θ_SFT20k + α × Σ_i (θ_PPOi - θ_SFT20k)
```
- α = 0.35（每路 task vector 的缩放系数）
- 融合参数范围：PPO 实际训练的 208/813 个 key（`PI05_PPO_TRAINABLE_PREFIXES`，action expert + projections）
- backbone（604 个 `paligemma.*` 参数）保持 SFT 原值不变

**脚本**：`scripts/task_arithmetic_rlinf.py`（基于 `openpi/scripts/task_arithmetic_pt.py` 改写，适配 `full_weights.pt` 格式）

### 10.3 权重路径

| 角色 | 路径 |
|------|------|
| Base（SFT 20k） | `logs/20260609-10:17:50-.../checkpoints/global_step_20000/actor/model_state_dict/full_weights.pt` |
| PPO adjust_bottle step-150 | `logs/20260610-10:56:32-.../checkpoints/global_step_150/actor/model_state_dict/full_weights.pt` |
| PPO place_container_plate step-150 | `logs/20260610-18:33:28-.../checkpoints/global_step_150/actor/model_state_dict/full_weights.pt` |
| PPO place_empty_cup step-150 | `logs/20260610-22:58:11-.../checkpoints/global_step_150/actor/model_state_dict/full_weights.pt` |
| **输出（merged）** | `logs/merged/sft20k_ppo150_adj_pcp_pec_a035/full_weights.pt` |

融合 Job：9777569，COMPLETED，208/813 个 key 改变，耗时 ~1 分钟（0 GPU）。

### 10.4 评测结果

| Task | SFT 20k（base，RLinf） | SFT 30k（RLinf） | Merged α=0.35（RLinf） | vs SFT 20k | vs SFT 30k |
|------|----------------------|-----------------|----------------------|------------|------------|
| adjust_bottle | 26.0% | 21.9% | **35.4%** | +9.4 pp | +13.5 pp |
| place_container_plate | 27.1% | 29.2% | **45.8%** | +18.7 pp | +16.6 pp |
| place_empty_cup | 28.1% | 30.2% | **47.9%** | +19.8 pp | +17.7 pp |
| **平均** | **27.1%** | **27.1%** | **43.0%** | **+16.0 pp** | **+15.9 pp** |

评测 Jobs：9777588（adjust_bottle）、9777589（place_container_plate）、9777590（place_empty_cup），均 COMPLETED。
SFT 20k Jobs：9777843、9777844、9777845，均 COMPLETED。

### 10.5 说明

- SFT 20k 与 SFT 30k 结果基本相近（相差 ≤3pp），说明 SFT 20k→30k 阶段对这三个任务的提升有限
- Merged α=0.35 相对 SFT 20k base 平均提升 **+16 pp**，三个任务一致
- 三路 task vector 每路 α=0.35，合计相当于 α×3=1.05 倍的平均 task vector，适度超出单任务 α=1 等价点
- 该方法只需一份 merged 权重，可同时在三个任务上获得类似单任务 PPO 的提升效果

---

## 附录：相关文档

- `my_docs/2026-6-8-experient-ppo.md`：SFT vs PPO 权重差异详解（value head、strict=False、critic_warmup_steps 原理）
- `my_docs/2026-6-7-experient-sft.md`：norm_stats 计算记录
- `/mnt/petrelfs/muyao/jiayusun/RoboTwin/my_docs/pi05_robotwin47_experiment_log.md`：47 任务 SFT baseline 成功率完整表
