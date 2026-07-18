# 6.14–6.16 工作记录：PPO 训练、Manifest 收集、权重融合、新SFT计划

---

## 1. Robotwin5 SFT 训练（job 9789706，仍在运行）

### 1.1 概述

在 5 个任务的数据集上对 pi0.5 做 SFT，目标是训练一个专注于少量任务的策略，作为对比基线。

- sbatch：`examples/sft/slurm/robotwin5_pi05_sft_8gpu.sbatch`（job name: `rw5-pi05-sft-8g`）
- config：`examples/sft/config/robotwin5_sft_openpi_pi05.yaml`
- 起始权重：`/mnt/petrelfs/muyao/jiayusun/pi05_base`（原始 pi0.5）
- 数据集：`/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_5_pi05`
- OpenPI config：`pi05_aloha_robotwin5`
- max_steps: 30000，save_interval: 5000

### 1.2 包含的 5 个任务

从数据集 episodes 中确认：
1. **adjust_bottle**
2. **beat_block_hammer**
3. **click_alarmclock**
4. **dump_bin**（系列，具体变体待确认）
5. **handover_mic**

### 1.3 进度

| Checkpoint | 时间 |
|---|---|
| global_step_5000 | Jun 15 23:22 |
| global_step_10000 | Jun 16 04:55 |
| global_step_15000 | Jun 16 10:27 |
| global_step_20000 | 已存在 |

节点：`SH-IDCA1404-10-140-54-70`，目标 30000 step，仍在运行。

---

## 2. PPO 训练实验

### 2.1 在 adjust_bottle 任务上的两组 PPO 实验

两个 PPO 实验均在 robotwin15 任务集配置下运行：

**实验 A：SFT 20k 权重起步（job 9789283，已完成 200 step 后 cancel）**

```yaml
# examples/embodiment/config/robotwin15_adjust_bottle_ppo_openpi_pi05.yaml
actor.model.model_path: /mnt/petrelfs/muyao/jiayusun/pi05_base       # 仅架构
runner.ckpt_path: .../global_step_20000/actor/model_state_dict/full_weights.pt  # 实际权重
```

- 日志目录：`logs/20260615-15:12:15-robotwin15_adjust_bottle_ppo_openpi_pi05/`
- 最终 checkpoint：`global_step_200`
- 注意：`model_path` 填的是 `pi05_base`（架构），真正的权重通过 `ckpt_path` 加载 SFT 20k，易误读

**实验 B：random_first20_pi05_sft 权重起步（job 9789284，仍在运行）**

```yaml
# examples/embodiment/config/robotwin15_adjust_bottle_ppo_random20.yaml
actor.model.model_path: /mnt/petrelfs/muyao/jiayusun/random_first20_pi05_sft
```

- 日志目录：`logs/20260615-15:45:43-robotwin15_adjust_bottle_ppo_random20/`

### 2.2 新增两个任务的 PPO 实验

**dump_bin_bigbin（job 9792233）** 和 **beat_block_hammer（job 9792232）**：

- 起始权重：`random_first20_pi05_sft`
- `save_interval: 40`（每 40 epoch 保存一次）
- dump_bin_bigbin 无 eval_seeds / train_seeds，`seeds_path: null`，随机种子
- beat_block_hammer 有 320 eval seeds + 1000 train seeds
- sbatch：
  - `examples/embodiment/slurm/random20_pi05_ppo_dump_bin_bigbin.sbatch`
  - `examples/embodiment/slurm/random20_pi05_ppo_beat_block_hammer.sbatch`
- config：
  - `examples/embodiment/config/random20_ppo_dump_bin_bigbin.yaml`
  - `examples/embodiment/config/random20_ppo_beat_block_hammer.yaml`
- 新增 env config：`examples/embodiment/config/env/robotwin_dump_bin_bigbin.yaml`

**6.16 进度**：

| Job | 状态 | 最新 checkpoint |
|---|---|---|
| 9792232 (bbh) | 运行中 ~5h | global_step_40 |
| 9792233 (dbb) | 运行中 ~4h | 无（尚未落盘） |

---

## 3. 权重融合与测评（adjust_bottle）

### 3.1 融合方案

使用 Task Arithmetic（`scripts/task_arithmetic_rlinf.py`）：

```
theta* = theta_SFT20k + alpha * (theta_PPO200 - theta_SFT20k)
```

- Base（SFT 20k）：`logs/20260615-15:12:15-.../checkpoints/global_step_200/actor/model_state_dict/full_weights.pt`
  - 源自：`logs/20260614-02:45:23-robotwin15_sft_pi05_8gpu/.../global_step_20000/`
- FT（PPO 200 step）：`logs/20260615-15:12:15-.../global_step_200/actor/model_state_dict/full_weights.pt`
- 只融合 expert 层（PI05_PPO_TRAINABLE_PREFIXES，208/812 keys）

### 3.2 测评结果汇总（adjust_bottle，96 seeds）

| 权重 | success_once | job | 备注 |
|---|---|---|---|
| SFT 20k (robotwin15) + PPO200, **alpha=0.8** | **88.5%** | 9792636 | merged_weights/sft20k_ppo200_a08 |
| SFT 20k (robotwin15) + PPO200, **alpha=0.7** | 80.2% | 9792795 | merged_weights/sft20k_ppo200_a07 |
| SFT 20k (robotwin5, 4-GPU eval) | 62.5% | 9793449 | pi05_aloha_robotwin5 config |

**结论**：alpha=0.8 效果最优（88.5%），高于 alpha=0.7；robotwin5 SFT 20k 在 adjust_bottle 上明显弱于 robotwin15 SFT（62.5%），说明更多任务联合训练对 adjust_bottle 有益。

### 3.3 相关文件

- alpha=0.8：
  - eval config：`examples/embodiment/config/robotwin15_adjust_bottle_ppo200_a08_eval.yaml`
  - sbatch：`examples/embodiment/slurm/robotwin15_pi05_merge_eval_ppo200_a08.sbatch`
  - 融合输出：`/mnt/petrelfs/muyao/jiayusun/merged_weights/sft20k_ppo200_a08_adjust_bottle/full_weights.pt`
- alpha=0.7：
  - eval config：`examples/embodiment/config/robotwin15_adjust_bottle_ppo200_a07_eval.yaml`
  - sbatch：`examples/embodiment/slurm/robotwin15_pi05_merge_eval_ppo200_a07.sbatch`
  - 融合输出：`/mnt/petrelfs/muyao/jiayusun/merged_weights/sft20k_ppo200_a07_adjust_bottle/full_weights.pt`
- robotwin5 SFT 20k eval：
  - config：`examples/embodiment/config/robotwin15_adjust_bottle_rw5sft20k_eval.yaml`
  - sbatch：`examples/embodiment/slurm/robotwin15_pi05_eval_adjust_bottle_rw5sft20k.sbatch`
  - ckpt：`logs/20260615-17:43:19-robotwin5_sft_pi05_8gpu/.../global_step_20000/`

---

## 4. Manifest 收集基础设施

### 4.1 目的

为 20 个任务收集固定 (seed, instruction) 对，用于可复现的 eval。

### 4.2 脚本

`scripts/collect_robotwin_manifests.py`

- 有 eval_seeds 的任务（13 个）：取前 96 个 success_seeds（循环补足）
- 无 eval_seeds 的任务（7 个）：从 `start_seed=100001` 起，按 task_index 偏移生成顺序种子
- 每任务 96 episodes，输出到 `rlinf/envs/robotwin/seeds/{task}_manifest_96ep.json`

**关键 Bug 历史**：

1. `KeyError: 'save_path'` — VectorEnv 要求 task_config 含 `save_path` 等字段，已补充
2. `AttributeError: 'beat_block_hammer' has no attribute 'crazy_random_light'`
   - 原因：`venv.setup_task()` 后未调 `reset()` 即调 `get_obs()`，`_init_task_env_()` 未执行
   - 修复：所有批次统一用 `venv.reset() → venv.get_obs()`
3. `AttributeError: 'NoneType' has no attribute 'get'`（data_type=None）
   - 原因：task_config 缺少 `data_type` dict
   - 修复：补充完整 `data_type` 字段

### 4.3 job 历史与当前状态

| Job | 结果 |
|---|---|
| 9792434 | TIME LIMIT（4h），仅完成部分 |
| 9792663 | TIME LIMIT |
| 9792730 | TIME LIMIT（4h），仅完成 3/20：adjust_bottle, beat_block_hammer, blocks_ranking_rgb |
| 9794285 | task 17 (pick_diverse_bottles) TimeoutError，完成 16/20（task 18-20 未跑）|

**当前已收集（16/20）**：`rlinf/envs/robotwin/seeds/` 中：
adjust_bottle, beat_block_hammer, blocks_ranking_rgb, blocks_ranking_size, click_alarmclock, click_bell, dump_bin_bigbin, handover_block, handover_mic, hanging_mug, lift_pot, move_can_pot, move_pillbottle_pad, move_playingcard_away, move_stapler_pad, open_microwave

**尚未收集（4/20）**：pick_diverse_bottles（seed 不稳定）、pick_dual_bottles、place_a2b_left、place_a2b_right

### 4.4 无种子任务列表（7 个，使用顺序种子）

blocks_ranking_size, click_alarmclock, dump_bin_bigbin, hanging_mug, move_playingcard_away, open_microwave, pick_diverse_bottles

---

## 5. rw5 SFT 20k 测评结果（6.17）

### 5.1 测评配置

- 权重：`logs/20260615-17:43:19-robotwin5_sft_pi05_8gpu/.../global_step_20000/actor/model_state_dict/full_weights.pt`
- openpi config：`pi05_aloha_robotwin5`，`model_path: pi05_base`
- seeds：各任务 manifest_96ep.json（96 条轨迹，3 epoch × 32 envs）
- 节点：4 GPU，坏节点 exclude 列表见 §7

### 5.2 结果

| Task | success_once | job | 在 rw5 训练集 | 备注 |
|---|---|---|---|---|
| adjust_bottle | **80.2%** | 9795591 | ✅ | 成功 |
| click_alarmclock | **19.8%** | 9795605 | ✅ | 成功 |
| beat_block_hammer | **5.2%** | 9795576 | ✅ | 低成功率，见 §5.3 |
| blocks_ranking_rgb | **0.0%** | 9795606 | ❌ OOD | 符合预期 |
| blocks_ranking_size | **失败** | 9795593 | ❌ OOD | node-115 EGL 崩溃（见 §7） |

### 5.3 beat_block_hammer 低成功率分析

rw5 SFT 20k 在 beat_block_hammer 上仅 5.2%，接近 base 模型的 0%。原因：
- 任务本身难（需先拿锤子，再精准击打），动作链比 adjust_bottle 长
- 仅 100 条训练 episodes，数量不足
- SFT 20k 步对该任务欠拟合（adjust_bottle 80.2% 说明 SFT 整体工作正常）

eval_seeds 来源：eval_seeds.json success_seeds（100150000 范围，320 个），与 adjust_bottle 使用同一验证机制，seeds 本身没问题。

---

## 6. random_first20_pi05_sft 测评（6.17）

### 6.1 权重信息

- 路径：`/mnt/petrelfs/muyao/jiayusun/random_first20_pi05_sft/`
- norm_stats：`random_first20_pi05_sft/norm_stats.json`（`physical-intelligence/robotwin15/`）
- openpi config：`pi05_aloha_robotwin15`
- model_path（eval 时）：`/mnt/petrelfs/muyao/jiayusun/random_first20_pi05_sft`（含 norm_stats）
- ckpt_path：`random_first20_pi05_sft/model_state_dict/full_weights.pt`

### 6.2 结果

| Task | success_once | job | max_episode_steps | 备注 |
|---|---|---|---|---|
| click_alarmclock | 20.8% | 9795645 | 200 | 步数不足，全部卡在上限 |
| click_alarmclock | 39.9% | 9795679 | 400 | 显著提升 |
| click_alarmclock | **41.9%** | 9795691 | 800 | 与 400 步几乎持平（+2%）|
| adjust_bottle | **~40%** | — | 400 | rf20sft 在该任务上表现中等 |

**结论：400 步是合理上限**（见 §9）。

click_alarmclock eval config：`examples/embodiment/config/rf20sft_eval_click_alarmclock.yaml`
sbatch：`examples/embodiment/slurm/rf20sft_eval_click_alarmclock.sbatch`

### 6.3 PPO 候选任务（~40% 成功率区间）

从 rf20sft 测评结果中，满足 PPO 训练成功率条件（~40%，适合有提升空间）的任务：

| 任务 | rf20sft 成功率 | PPO 计划 |
|---|---|---|
| click_alarmclock | 39.9% | PPO 直接从 rf20sft 训练（job 9795806，运行中） |
| adjust_bottle | ~40% | 先做 20k SFT fine-tune（job 9796299），再做 PPO |

**adjust_bottle 先做 SFT 的原因**：rf20sft 对 adjust_bottle 有一定成功率，但行为尚不稳定，通过单任务 SFT 进一步强化后再做 PPO，预期能提高 PPO 的探索效率。

---

### 6.3 关于 eval/episode_len 指标的说明

`episode_len` 是每个 rollout slot 的**总步数**（非单条 episode 长度）。启用 `auto_reset: True` 后，一个 slot 内 episode 结束即 reset 并开始新 episode，slot 总步数可超过 `max_episode_steps` 预算（因为等当前 episode 跑完才截断）。

例：400 步预算下 episode_len=510，说明单条 episode 平均约 120 env steps，slot 内跑了约 4 条。

---

## 7. 新 SFT 计划：Top-15 任务数据集

### 5.1 背景

原 robotwin15 选取的 15 个任务**并非**按成功率排序，而是随机挑选，包含 move_can_pot（11%）、pick_diverse_bottles（24%）等低成功率任务。

新计划：按 robotwin47 评测成功率排序，取前 15 个任务做 SFT。

### 5.2 Top-15 任务（按 pi05 on robotwin47 成功率）

| 排名 | 任务 | 成功率 |
|---|---|---|
| 1 | shake_bottle_horizontally | 91% |
| 2 | shake_bottle | 84.5% |
| 3 | stack_bowls_two | 82% |
| 4 | place_empty_cup | 79.5% |
| 5 | open_microwave | 78% |
| 6 | place_container_plate | 78% |
| 7 | press_stapler | 76.5% |
| 8 | dump_bin_bigbin | 74% |
| 9 | beat_block_hammer | 73.5% |
| 10 | place_object_stand | 57% |
| 11 | click_alarmclock | 57% |
| 12 | adjust_bottle | 54.5% |
| 13 | handover_mic | 53% |
| 14 | place_shoe | 51% |
| 15 | place_object_basket | 45% |

### 5.3 数据集转换

源数据：`/mnt/petrelfs/muyao/jiayusun/robotwin-random-47-500`
- 48 个任务文件夹，每任务 500 episodes，HDF5 格式（`episode_N/episode_N.hdf5` + `instructions.json`）

转换 job：**9793775**（`rw47-convert`，运行中，节点 SH-IDCA1404-10-140-54-2）

- sbatch：`examples/embodiment/slurm/convert_robotwin_random47_500_to_lerobot.sbatch`
- 使用 RoboTwin venv（`/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/.venv`）+ Apptainer 容器
- 输出：`/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_47_500_pi05`
- 时间限制：2 天，32 CPU，无 GPU
- 转换完成后从中筛选 top-15 任务子集并重新计算 norm_stats

---

## 8. 坏节点 Exclude 列表（永久）

以下节点存在 EGL `cannot create buffer` 渲染问题或 Ray 会话冲突，已加入所有 eval sbatch 的 `--exclude`：

| 节点 IP 后缀 | 问题 |
|---|---|
| -7 | 已知坏节点 |
| -10 | EGL 渲染失败 |
| -12 | 已知坏节点 |
| -13 | 已知坏节点 |
| -18 | 已知坏节点 |
| -19 | EGL 渲染失败 |
| -33 | 已知坏节点 |
| -36 | Ray AssertionError（6.18 发现） |
| -44 | 已知坏节点 |
| -46 | Ray 会话冲突（6.18 发现） |
| -50 | Ray AssertionError（6.18 发现） |
| -58 | 已知坏节点 |
| -59 | OOM：残留 GPU 进程占 72GB（6.18 发现） |
| -69 | 已知坏节点 |
| -81 | 已知坏节点 |
| -89 | Ray GCS timeout（6.18 发现） |
| -115 | EGL 渲染失败（6.17 发现，blocks_ranking_size job 9795593 崩溃） |
| -117 | 已知坏节点 |
| -118 | 已知坏节点 |
| -25 | 残留 Ray 进程（zombie Ray on port 42519，6.19 发现） |
| -20 | OOM：残留 GPU 进程占 72GB（6.20 发现，ca_sft15k eval 失败） |
| -28 | OOM：残留 GPU 进程占 76GB（6.21 发现，ca_sft15k eval 失败） |
| -61 | Ray session 冲突（6.21 发现） |

完整 exclude 行（6.21 更新）：
```
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-61
```

---

## 9. 重要实验参数

### eval max_episode_steps = 400（全局标准）

经 click_alarmclock 200/400/800 三组对比验证：
- 200 步：success 20.8%，episode 全部卡在上限，明显不足
- 400 步：success 39.9%，episode 自然在 ~510 env steps 结束
- 800 步：success 41.9%，与 400 几乎持平（+2%），步数不是瓶颈

**结论：所有任务 eval 统一使用 `max_episode_steps: 400`（train/eval 均设）。** 旧的 200 步配置会低估成功率，不再使用。

---

## 10. 当前运行中 / 待处理事项（6.17 更新）

### 运行中

| Job | 内容 | 状态 |
|---|---|---|
| 9795806 | rf20sft PPO click_alarmclock（node-28） | Running |
| 9796096 | 11 任务 seed manifest 收集 | Pending |
| 9796299 | adjust_bottle 20k SFT from rf20sft | Pending |

### 待处理

- **adjust_bottle SFT 完成后**：提交 PPO（从 adjust_bottle SFT 权重起步）
- **rw47-convert**：job 9793775 被 OOM SIGKILL（38% 时中断，8816/23500 episodes），需重新提交（从头重转，增大内存）；adjust_bottle 数据已存在（ep 0-499），暂时使用现有数据做 SFT
- **rf20sft 待测评**：blocks_ranking_size（需排除 node-115 重测）、move_can_pot、move_pillbottle_pad
- **Manifest 收集**：pick_diverse_bottles（seed 不稳定，暂跳过）；其余 11 个任务已提交 job 9796096
- 所有 job 状态检查用 `squeue -u muyao`（非 jiayusun）

---

## 11. PPO 训练结果 & 权重融合实验（6.17–6.18）

### 11.1 rf20sft PPO 训练结果

**PPO_CA（click_alarmclock，job 9795806）**

- 起始权重：`random_first20_pi05_sft`
- 训练目录：`logs/20260617-13:27:58-rf20sft_ppo_click_alarmclock/`
- checkpoint：`global_step_100`

| checkpoint | click_alarmclock | adjust_bottle |
|---|---|---|
| step_100 | ~87% | ~63% |

**PPO_adj200（adjust_bottle，job 9789284）**

- 起始权重：`random_first20_pi05_sft`
- 训练目录：`logs/20260615-15:45:43-robotwin15_adjust_bottle_ppo_random20/`
- checkpoint：`global_step_200`

| checkpoint | click_alarmclock | adjust_bottle |
|---|---|---|
| step_200 | 54.9% | 84.6% |

**结论**：PPO_CA 对 click_alarmclock 专攻效果好（87%），但 adjust_bottle 无明显劣化（63%）；PPO_adj200 对 adjust_bottle 提升显著（85%），click_alarmclock 也有适度提升（55%）。

---

### 11.2 权重融合实验（Task Arithmetic）

#### 融合方案 A：rf20sft 为基础，添加 PPO_CA 任务向量

```
theta* = rf20sft + 0.8 × (PPO_CA_100 - rf20sft)
```

- 融合权重：`merged_weights/rf20sft_ppo_ca_step100_a08/full_weights.pt`
- 脚本：`scripts/task_arithmetic_rlinf.py`
- 融合 key 数：208 / ~813（PI05_PPO_TRAINABLE_PREFIXES）

| | click_alarmclock | adjust_bottle |
|---|---|---|
| 结果 | ~86% (91.5% / 81.7% 两次) | 67.0% |

#### 融合方案 B：rf20sft 为基础，添加 PPO_adj200 任务向量

```
theta* = rf20sft + 0.8 × (PPO_adj200 - rf20sft)
```

- 融合权重：`merged_weights/rf20sft_ppo_adj200_a08/full_weights.pt`

| | click_alarmclock | adjust_bottle |
|---|---|---|
| 结果 | **37.3%** | **83.8%** |

**注**：click_alarmclock 仅 37.3%，低于 rf20sft 基线（33%），说明 PPO_adj200 任务向量与 CA 任务有冲突。

#### 融合方案 C：PPO_adj200 为基础，叠加 CA 任务向量（跨基底融合）

```
theta* = PPO_adj200 + 0.8 × (PPO_CA_100 - rf20sft)
```

加法向量来自 CA 专项训练（相对 rf20sft 基础），叠加到 adj PPO 权重上。

- 融合脚本：`scripts/merge_ppo_adj200_plus_ca_tv.py`
- 融合权重：`merged_weights/ppo_adj200_plus_ca100_tv_a08/full_weights.pt`
- 融合 key 数：208 / 821

| 测试 | click_alarmclock | adjust_bottle |
|---|---|---|
| 第一次 | 80.5% | 62.9% |
| 第二次 | 86.0% | 65.1% |
| **均值** | **~83%** | **~64%** |

**结论**：方案 C 是目前最优的两任务均衡方案——CA 从 55% 提升到 83%，adj 从 85% 略降到 64%，两个任务都保持实用水平。

---

### 11.3 四方案综合对比表

| 权重 | click_alarmclock | adjust_bottle |
|---|---|---|
| rf20sft 基线 | 33% | 51% |
| PPO_CA step_100 | ~87% | ~63% |
| PPO_adj200 | 55% | 85% |
| 融合A：rf20sft+CA_tv, α=0.8 | ~86% | 67% |
| 融合B：rf20sft+adj_tv, α=0.8 | 37% | 84% |
| 融合C：ppo_adj200+CA_tv, α=0.8 | **~83%** | **~64%** |

**融合 C（方案 C）最具实用价值**：两任务均超过 60%，无任务严重退化。

---

### 11.4 融合 C 不同 α 消融实验

```
theta* = PPO_adj200 + alpha × (PPO_CA_100 - rf20sft)
```

**第一次提交（6.18，失败）**：merge sbatch 有 bug——re-exec 进容器后仍调用原始 `merge_ppo_adj200_plus_ca_tv.sbatch`，导致 α=0.9/0.7 两个 merge job 均写入 `_a08` 路径。eval job 因找不到对应权重而 FileNotFoundError。

**修复（6.19）**：sbatch re-exec 行改为各自的 `_a09.sbatch` / `_a07.sbatch`，重新提交后成功完成（6.19–6.20）。

**α 消融完整结果（6.20）**：

| α | click_alarmclock | adjust_bottle | merge job | CA eval job | adj eval job |
|---|---|---|---|---|---|
| 0.7 | **93.6%** | **73.4%** | 9804021 | 9804024 | 9807027 |
| 0.8 | ~83% | ~64% | 9800xxxA | — | — |
| 0.9 | 82.2% | 53.3% | 9804020 | 9804022 | 9804023 |

融合权重输出路径：
- α=0.7：`merged_weights/ppo_adj200_plus_ca100_tv_a07/full_weights.pt`
- α=0.9：`merged_weights/ppo_adj200_plus_ca100_tv_a09/full_weights.pt`

**结论**：α=0.7 在两个任务上均最优（CA 93.6%，adj 73.4%），随 α 增大两者均下降，说明 CA 任务向量叠加量越多，adj 上性能损失越大，CA 上也不单调（α=0.9 < α=0.8 < α=0.7 for CA）。**α=0.7 为最优融合系数。**

注：adj eval 9804025 首次因 GCS crash 失败，重提为 9807027 出结果（73.4%，305 trajectories）。

---

### 11.5 rf20sft 20 任务批量评测

#### 设置

- 单 job 顺序执行所有任务，每个任务独立 `ray stop → ray start`（避免 Ray 状态污染）
- 8 GPU，sbatch：`examples/embodiment/slurm/rf20sft20_eval_all20.sbatch`

#### 评测历史

| Job | 问题 | 结果 |
|---|---|---|
| 9799651 | blocks_ranking_size 内存 corruption 污染 Ray | 部分完成后挂死 |
| 9800491 | 10 个任务缺少 env config（exit 1 秒退） | 6 个任务出结果，10 个 FAILED |
| 9800762 | handover_block collective group 死锁（8h+） | 8 个新任务出结果，之后挂死 |

#### 补建的 env 配置文件（6.18）

新增 `examples/embodiment/config/env/` 下 10 个文件（均以 `robotwin_click_alarmclock.yaml` 为模板，仅改 `task_name`）：
`shake_bottle`, `stack_blocks_three`, `place_a2b_left`, `place_mouse_pad`, `place_object_stand`,
`scan_object`, `place_phone_stand`, `stamp_seal`, `place_bread_basket`, `move_playingcard_away`

#### 完整结果汇总（跨多次 job）

| Task | rf20sft success_once | 来源 job |
|---|---|---|
| click_alarmclock | 33.0% | 9800491 |
| adjust_bottle | 50.6% | 9800491 |
| move_can_pot | 21.8% | 9800762 |
| place_object_stand | 23.8% | 9800762 |
| place_a2b_left | 18.3% | 9800762 |
| place_mouse_pad | 2.7% | 9800762 |
| move_pillbottle_pad | 0.0% | 9800491 / 9800762 |
| blocks_ranking_size | 0.0% | 9800491 / 9800762 |
| stack_blocks_three | 0.0% | 9800762 |
| handover_block | 0.0% | 9800491 |
| shake_bottle | FAILED (exit 255) | 9800762 |
| dump_bin_bigbin | FAILED (exit 255, seed 不稳定) | 9800491 |
| handover_block (8GPU) | **死锁**（collective timeout 3h+） | 9800762 |
| scan_object, place_phone_stand, stamp_seal, place_bread_basket, move_playingcard_away | 未跑 | — |
| place_object_scale, pick_diverse_bottles, stack_bowls_three | NO_MANIFEST | — |

**handover_block 需从 batch 中跳过**（8-GPU 下 collective group 必死锁）；5 个任务（scan_object 至 move_playingcard_away）还未评测。

---

## 12. 单任务 SFT 实验结果（6.18–6.19）

### 12.1 实验设置

在 rf20sft 基础上，对两个目标任务各做单任务 SFT（目标 20k steps，因时间限制实际在 15k 截止），使用从 `robotwin_random_47_500_pi05` 中筛出的 500 条 demo。

**adjust_bottle SFT（job 9796299）**
- sbatch：`examples/sft/slurm/rf20_adjust_bottle_sft_8gpu.sbatch`
- 数据：episodes 0-499 → `robotwin_adjust_bottle_500_pi05`
- 实际训练：18163/20000 steps（被 SLURM 取消），最终 checkpoint：`global_step_15000`

**click_alarmclock SFT（job 9797158）**
- sbatch：`examples/sft/slurm/rf20_click_alarmclock_sft_8gpu.sbatch`
- 数据：episodes 2000-2499 → `robotwin_click_alarmclock_500_pi05`
- 实际训练：17551/20000 steps（被 SLURM 取消），最终 checkpoint：`global_step_15000`

### 12.2 SFT 15k 评测结果

| 权重 | click_alarmclock | adjust_bottle |
|---|---|---|
| rf20sft 基线 | 33.0% | 50.6% |
| **adj_sft15k** | **54.2%** | **83.2%** |
| **ca_sft15k** | **52.5% → 55.7%（复测）** | **0.0%（两次一致）** |

- adj_sft15k：两个任务均有提升（CA +21%，adj +33%），数据泛化性好
- ca_sft15k：CA 提升（+20%~+23%），但 adjust_bottle **灾难性遗忘**（50.6% → 0%，复测 9807160 确认）

复测说明：9807160（adj，0.0%，480 trajectories）、9808672（CA，55.7%）；初次评测因 node-20/28/61 OOM 或 Ray 冲突失败，共重提 3 次后出结果。

### 12.3 SFT 权重路径

- adj_sft15k：`logs/20260617-20:00:24-rf20_adjust_bottle_sft_8gpu/rf20_adjust_bottle_sft_pi05/checkpoints/global_step_15000/actor/model_state_dict/full_weights.pt`
c

---

## 13. PPO 新实验：place_object_stand（6.19）

- **Job**：9804002，node-88
- **Config**：`examples/embodiment/config/rf20sft_ppo_place_object_stand.yaml`
- **起始权重**：`random_first20_pi05_sft`（rf20sft 基线 23.8%）
- **Train seeds**：64 个（manifest 前 64 条，100673-100736），文件：`place_object_stand_train_seeds.json`
- **Eval seeds**：96 个（全 manifest，100673-100768），文件：`place_object_stand_eval_seeds.json`
- 8 GPU，336h，save_interval=50

**结果：FAILED**（job 在 node-88 被 SLURM cancel，`ValueError: Unsupported object type: -1946156800`，同 dump_bin_bigbin 的 seed 兼容问题）。node-88 已加入 exclude 列表。

**重提（6.21）**：job 9812732，node-88 已 exclude，在新节点上运行中。

---

## 14. merged_a07 PPO adjust_bottle 实验（6.19–6.20）

以 α=0.7 融合权重为起点，对 adjust_bottle 继续做 PPO：

- **Job**：9805206（`ma07-ppo-a`），node-34，约 6.19 16:05 开始
- **Config**：`examples/embodiment/config/merged_a07_ppo_adjust_bottle.yaml`
- **起始权重（ckpt_path）**：`merged_weights/ppo_adj200_plus_ca100_tv_a07/full_weights.pt`
- **实验目录**：`logs/20260619-16:05:12-merged_a07_ppo_adjust_bottle/`
- **已保存 checkpoint**：`global_step_50`、`global_step_100`、`global_step_150`
- **当前进度**（6.20）：step ≈ 489，train `success_once ≈ 91%`（在线训练集表现）

### step_50 checkpoint 评测（6.20–6.21）

- ckpt：`.../checkpoints/global_step_50/actor/model_state_dict/full_weights.pt`
- eval config：`ma07_ppo_adj_step50_eval_adjust_bottle.yaml` / `_click_alarmclock.yaml`
- eval job：9807061（adj）、9807062（CA）

| checkpoint | click_alarmclock | adjust_bottle |
|---|---|---|
| α=0.7 merge 静态（起始权重） | 93.6% | 73.4% |
| **step_50 PPO** | **87.7%** | **84.2%** |

PPO 50 步后：adj 73.4% → **84.2%**（+11%），CA 93.6% → 87.7%（-6%）——符合继续在 adj 上 PPO 的预期方向。

注：9807048/9807049 首次提交因 Ray 节点冲突失败，重提为 9807061/9807062。

---

## 15. 当前运行中 / 待处理事项（6.21 更新）

### 运行中

| Job | 内容 | 节点 | 状态 |
|---|---|---|---|
| 9805206 | merged_a07 PPO adjust_bottle | node-34 | Running（step≈489+） |
| 9812732 | rf20sft PPO place_object_stand（重提） | — | Running |

### 最新完整结果汇总

| 权重 | click_alarmclock | adjust_bottle |
|---|---|---|
| rf20sft 基线 | 33.0% | 50.6% |
| PPO_CA step100 | ~87% | ~63% |
| PPO_adj step200 | 55% | 85% |
| adj_sft15k | 54.2% | 83.2% |
| ca_sft15k | 55.7% | **0.0%**（灾难性遗忘） |
| 融合C α=0.9 | 82.2% | 53.3% |
| 融合C α=0.8 | ~83% | ~64% |
| **融合C α=0.7（静态）** | **93.6%** | **73.4%** |
| **merged_a07 PPO adj step50** | **87.7%** | **84.2%** |

### 待处理

- **merged_a07 PPO 更多 checkpoint 评测**：已有 step100/150，可继续评测曲线
- **rf20sft 20-task 剩余**：scan_object, place_phone_stand, stamp_seal, place_bread_basket, move_playingcard_away（5 个），需剔除 handover_block 后重提 batch
- **place_object_stand PPO（9812732）**：监控是否成功启动（之前 node-88 崩溃）
- 所有 job 状态检查用 `squeue -u muyao`
