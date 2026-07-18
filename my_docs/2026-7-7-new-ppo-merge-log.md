# π0.5 (MoE) 新一轮 PPO 训练 + 融合实验记录

**开始日期**：2026-07-07  
**基础模型**：`pi05_aloha_moe_demo_clean` / SFT-20k checkpoint  
**关键变更**：`total_num_envs: 64 → 128`，`global_batch_size: 2048 → 4096`

---

## 1. 当前训练任务

| 任务 | Job ID | 当前 Step | max_steps | total_num_envs | 节点 |
|------|--------|----------|-----------|----------------|------|
| beat_block_hammer | 10204943 | 70/100 | 100 | 128 | -1 |
| place_a2b_left | 10204898 | 69/500 | 500 | 128 | -2 |
| move_can_pot | 10204899 | 65/500 | 500 | 128 | -24 |
| move_playingcard_away | 10202481 | 174/500 | 500 | 64（旧配置）| -25 |

> place_a2b_right (10200922) 已手动暂停，step200 checkpoint 可用。

---

## 2. 有 step50 权重的任务

| 任务 | 来源 log 目录 | 可用 checkpoints | 配置 |
|------|-------------|-----------------|------|
| beat_block_hammer | `20260706-23:57:25-moe_sft20k_beat_block_hammer_ppo100` | step50 | 128 envs |
| place_a2b_left | `20260706-23:28:22-moe_sft20k_place_a2b_left_ppo100` | step50 | 128 envs |
| place_a2b_right | `20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100` | step50, step200 | 64 envs（旧） |
| move_can_pot | `20260706-23:28:23-moe_sft20k_move_can_pot_ppo100` | step50 | 128 envs |
| move_playingcard_away | `20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100` | step50, step100, step150 | 64 envs（旧）|

**SFT-20k 单任务基线**（eval success_once，96ep manifest）：

| 任务 | SFT-20k | 旧 PPO 最优（单任务）|
|------|---------|-------------------|
| beat_block_hammer | 57.8% | 86.1% @step299 |
| place_a2b_left | 35.4% | 51.9% @step299 |
| place_a2b_right | 55.1% | 57.1% @step49 |
| move_can_pot | 52.6% | 63.6% @step399 |
| move_playingcard_away | 72.3% | — (新跑中) |

---

## 3. 融合实验记录（新 step50 权重，2026-07-07）

### 融合公式

```
theta* = theta_SFT20k + sum_i [ alpha_i * (theta_PPOi - theta_SFT20k) ]
```

融合范围：208 / 812 keys（`PI05_PPO_TRAINABLE_PREFIXES`）。  
脚本：`scripts/task_arithmetic_rlinf.py`（已支持 `--alphas` per-task 系数）。  
Eval：`eval_rollout_epoch: 3`（96ep × 3 = ~288ep 平均，减少方差）。

---

### 融合实验 1：3task_a07（bbh+pal+par，α=0.7 均匀）

**权重路径**：`/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_3task_bbh_pal_par_step50_a07/full_weights.pt`  
**系数**：bbh=0.7, pal=0.7, par=0.7  
**Eval Job**：10206391（par 补跑）+ 10206142（bbh/pal，已完成）

| 任务 | 单任务 PPO | **融合结果** | Δ |
|------|-----------|------------|---|
| beat_block_hammer | 86.1% | 60.9% | −25.2pp |
| place_a2b_left | 51.9% | **71.0%** ⭐ | **+19.1pp** |
| place_a2b_right | 57.1% | **66.1%** | **+9.0pp** |

**结论**：pal 和 par 均超过单任务基线，bbh 因与 pal/par 冲突下降明显。

---

### 融合实验 2：3task_bbh09_pal07_par07（bbh=0.9，pal/par=0.7）

**权重路径**：`/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_3task_bbh09_pal07_par07_step50/full_weights.pt`  
**系数**：bbh=0.9, pal=0.7, par=0.7  
**Eval Job**：10206403

| 任务 | 单任务 PPO | 3task_a07 | 3task_a09 | Δ(vs a07) |
|------|-----------|-----------|------------|-----------|
| beat_block_hammer | 86.1% | 60.9% | **73.7%** | +12.8pp ↑ |
| place_a2b_left | 51.9% | **71.0%** | 55.5% | −15.5pp ↓ |
| place_a2b_right | 57.1% | 66.1% | 62.3% | −3.8pp ↓ |

**结论**：提高 bbh 系数有效提升 bbh（73.7%），但代价是 pal 从 71.0% 大幅下降至 55.5%。bbh 与 pal 存在强烈冲突。

---

### 融合实验 3：2task_pal07_par07（pal+par，α=0.7）

**权重路径**：`/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_2task_pal07_par07_step50/full_weights.pt`  
**系数**：pal=0.7, par=0.7  
**Eval Job**：10206602

| 任务 | 单任务 PPO | 旧 pal+par（旧ckpt）| **融合结果** | Δ(vs 单任务) |
|------|-----------|-------------------|------------|------------|
| place_a2b_left | 51.9% | 64.5% | **63.7%** | +11.8pp |
| place_a2b_right | 57.1% | 56.0% | **67.5%** | +10.4pp |

**结论**：两任务融合都超过单任务，但 pal 不如加入 bbh 的 3task_a07（63.7% vs 71.0%）。bbh 对 pal 有正向迁移效应，尽管同时损害了 bbh 自身。

---

## 4. 融合对比汇总

| 融合方案 | bbh | pal | par | 平均 |
|---------|-----|-----|-----|------|
| 单任务 PPO | 86.1% | 51.9% | 57.1% | 65.0% |
| **3task_a07** | 60.9% | **71.0%** | **66.1%** | **66.0%** |
| 3task_bbh09 | **73.7%** | 55.5% | 62.3% | 63.8% |
| 2task_pal+par | — | 63.7% | **67.5%** | 65.6% |
| 旧 5task_a07（参考）| 36.8% | 55.6% | 52.4% | — |

**最优方案**：`3task_a07`（α=0.7 均匀）平均 66.0%，且 pal+par 双双超过单任务基线，整体最平衡。

---

## 5. 网格搜索补充（bbh × pal × par，2026-07-08）

**脚本**：`scripts/merge_bbh_pal_par_s50_grid.py` 生成权重，`examples/embodiment/slurm/eval_bbh_pal_par_s50_grid.sbatch` 评测。
**系数范围**：bbh/pal/par 各自在 `{0.5, 0.7, 0.9}`（pal/par 只测了 0.5/0.7）取值，共探索 10 组（未覆盖 bbh07_pal07_par07 / bbh09_pal07_par07 两个点）。
**Eval jobs**：10209446 / 10209616 / 10209636 / 10209639 / 10209640 / 10209641 / 10209727。

> **注意**：本节权重文件已在 2026-07-09 做 5task 实验前的磁盘清理中删除（quota 接近上限），以下数值是从 SLURM 日志重新解析出来的，权重本身不可复现，仅供参考。若某组合重复出现在多个 job 里，取时间更晚的那次。

| bbh α | pal α | par α | bbh | pal | par | avg |
|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 0.5 | 78.5% | 49.8% | 71.4% | 66.6% |
| 0.5 | 0.5 | 0.7 | 75.1% | 64.1% | 66.9% | **68.7%** |
| 0.5 | 0.7 | 0.5 | 76.1% | 51.7% | 71.6% | 66.5% |
| 0.5 | 0.7 | 0.7 | 69.0% | 65.7% | 61.1% | 65.3% |
| 0.7 | 0.5 | 0.5 | 72.5% | 65.5% | 60.8% | 66.3% |
| 0.7 | 0.5 | 0.7 | 78.6% | 58.5% | 56.5% | 64.5% |
| 0.7 | 0.7 | 0.5 | 73.3% | 64.7% | 62.9% | 67.0% |
| 0.9 | 0.5 | 0.5 | 60.4% | 55.4% | 62.8% | 59.5% |
| 0.9 | 0.5 | 0.7 | 67.1% | 68.8% | 67.2% | 67.7% |
| 0.9 | 0.7 | 0.5 | 74.9% | 60.5% | 63.1% | 66.2% |

网格里平均最高的是 `bbh05_pal05_par07`（68.7%），比第 3 节里均匀 0.7 的 `3task_a07`（66.0%）还高，但没有单独看三个任务是否都超过单任务基线，只是粗略参考，值得注意的是这权重已经删了，如果要用得重新合并。

---

## 6. pal + par + move_playingcard_away 3-task 融合（2026-07-07）

跟第 3 节的 bbh/pal/par 是不同的任务组合，用的是 pal/par/mpa 各自 step50 checkpoint。

**Eval jobs**：10206779（pal07_par07_mpa07）、10206847 + 10207132 + 10207351（pal08_par08_mpa07，中途因超时补跑了 par/mpa）。
**权重路径**：
- `/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_3task_pal07_par07_mpa07_step50/full_weights.pt`
- `/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_3task_pal08_par08_mpa07_step50/full_weights.pt`

（这两份权重文件目前仍然存在，没有被清理）

| 系数 | place_a2b_left | place_a2b_right | move_playingcard_away |
|---|---|---|---|
| pal=0.7, par=0.7, mpa=0.7 | 57.6% | 64.8% | 81.2% |
| pal=0.8, par=0.8, mpa=0.7 | 61.1% | 63.0% | 83.1% |

对比单任务基线（pal=51.9%, par=57.1%, mpa 当时"新跑中"无基线）：两个系数点上 pal/par 都超过单任务基线，提高 pal/par 系数到 0.8 对三个任务都有小幅正向影响。

---

## 7. 下一步方向

- 等待 beat_block_hammer / place_a2b_left / move_can_pot 更多 step 落盘（step100/150/200），尝试更晚 checkpoint 的融合
- move_playingcard_away step150 已可用（train success ~93-97%），考虑加入融合测试
- 探索 3task_a07 基础上加入 move_can_pot 的 4task 融合
- 网格搜索里 `bbh05_pal05_par07` 平均分最高但权重已删除，如果要继续深挖需要重新合并
