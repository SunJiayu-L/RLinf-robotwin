# 五任务权重融合记录（6.7-6）

**任务**：beat_block_hammer, place_phone_stand, place_a2b_left, place_a2b_right, move_can_pot
**Base**：SFT-20k（`pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors`）
**系数**：alpha = 0.7（统一）

---

## 方案一：均匀多任务融合（Task Arithmetic，N=5）

```
theta* = theta_SFT20k + 0.7 * [ (PPO_bbh - theta_SFT20k)
                               + (PPO_pps - theta_SFT20k)
                               + (PPO_pal - theta_SFT20k)
                               + (PPO_par - theta_SFT20k)
                               + (PPO_mcp - theta_SFT20k) ]
```

- 5 个任务向量地位对等，各自系数均为 0.7，一次性求和后加到 base 上
- 脚本：`scripts/task_arithmetic_rlinf.py`（已扩展支持 `.safetensors` base 输入）
- 输出：`/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_sft20k_ppo5task_a07/full_weights.pt`

## 方案二：跨基底链式叠加（仿 doc1 方案 C，beat_block_hammer 为锚点）

```
theta1 = PPO_bbh                                   (bbh 系数 = 1，锚点任务)
theta2 = theta1 + 0.7 * (PPO_pps - theta_SFT20k)
theta3 = theta2 + 0.7 * (PPO_pal - theta_SFT20k)
theta4 = theta3 + 0.7 * (PPO_par - theta_SFT20k)
theta5 = theta4 + 0.7 * (PPO_mcp - theta_SFT20k)
```

- beat_block_hammer 以完整 PPO 权重作为起点（相当于系数 1），其余 4 个任务的 task vector（相对 SFT20k base）依次以 0.7 折入
- 结果依赖任务顺序，并非 5 个任务真正均等的 0.7——bbh 权重最高
- 脚本：`scripts/merge_moe_ppo5task_chain_a07.py`
- 输出：`/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_sft20k_ppo5task_chain_bbh_a07/full_weights.pt`

---

## 融合范围

只融合 PPO 实际训练（`train_expert_only=True`）改变的参数，即 `PI05_PPO_TRAINABLE_PREFIXES`：

```
paligemma_with_expert.gemma_expert.model.layers.
paligemma_with_expert.gemma_expert.model.norm.
action_in_proj.
action_out_proj.
time_mlp_in.
time_mlp_out.
```

两种方案均为 208 / 812 keys 参与融合。

## 使用的 PPO Checkpoint（均取各任务 eval 最优 step，见 `2026-7-5-ppo-sft20k-experiment-log.md`）

| 任务 | checkpoint | eval success_once |
|---|---|---|
| beat_block_hammer | `logs/20260625-11:37:02-moe_sft20k_beat_block_hammer_ppo100/.../checkpoints/global_step_300/actor/model_state_dict/full_weights.pt` | 86.1% @step299 |
| place_phone_stand | `logs/20260703-21:07:14-moe_sft20k_place_phone_stand_ppo100/.../checkpoints/global_step_300/actor/model_state_dict/full_weights.pt` | 70.7% @step299 |
| place_a2b_left | `logs/20260703-21:07:14-moe_sft20k_place_a2b_left_ppo100/.../checkpoints/global_step_300/actor/model_state_dict/full_weights.pt` | 51.9% @step299 |
| place_a2b_right | `logs/20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100/.../checkpoints/global_step_50/actor/model_state_dict/full_weights.pt` | 57.1% @step49 |
| move_can_pot | `logs/20260703-00:38:41-moe_sft20k_move_can_pot_ppo100/.../checkpoints/global_step_400/actor/model_state_dict/full_weights.pt` | 63.6% @step399 |

---

## 5 任务融合评测结果（6.7-6）

两种方案在全部 5 个任务上顺序评测（单 job 内逐任务 `ray stop`/`ray start`，避免同时提交 5 个任务）：

| Task | 单任务 PPO | 方案一：均匀融合 a07 | 方案二：链式融合(bbh锚点) |
|---|---|---|---|
| beat_block_hammer | 86.1% | 36.8% | 29.0% |
| place_phone_stand | 70.7% | 34.5% | 21.9% |
| place_a2b_left | 51.9% | **55.6%** | 32.9% |
| place_a2b_right | 57.1% | 52.4% | 36.1% |
| move_can_pot | 63.6% | 35.3% | 16.0% |

**结论**：均匀融合全面优于链式融合；链式方案让 bbh 锚定为系数 1 并未如预期保住 bbh 性能，反而因叠加了其余 4 个任务的完整 delta 而更差。5 任务同时融合时冲突效应明显，几乎所有任务都显著低于单任务 PPO。

sbatch：`examples/embodiment/slurm/moe_ppo5task_a07_eval_all5.sbatch`、`moe_ppo5task_chain_bbh_a07_eval_all5.sbatch`（job 10202685 / 10202686）

---

## 两两融合冲突性分析（10 对，alpha=0.7，N=2）

公式（与方案一相同，仅 N=2）：

```
theta* = theta_SFT20k + 0.7 * [ (PPO_i - theta_SFT20k) + (PPO_j - theta_SFT20k) ]
```

为了在存储受限（用户配额剩余仅 ~235GB）情况下完成全部 C(5,2)=10 组合，采用「merge → eval 两个任务 → 删除该对合并权重 → 下一对」的单 job 顺序流程，磁盘峰值只多占一对权重（~8GB）。

sbatch：`examples/embodiment/slurm/moe_ppo2task_a07_eval_pairs.sbatch`（job 10202794，完成前 8 对后在 pal+mcp 首次 eval 时遭遇已知的 Ray collective 通信崩溃 `ValueError: Unsupported object type` / `UnpicklingError`，卡死 6 分钟后判定为死锁，`scancel` 后用 `moe_ppo2task_a07_eval_pairs_remaining.sbatch`（job 10203831，排除故障节点-29）补跑剩余 2 对）。

前置：删除了 5 个任务中已分析完毕的非最优 PPO 中间 checkpoint（只保留融合用到的最优 step），释放约 500GB（15.92 TiB → 15.42 TiB）。

### 完整结果（单任务基线：bbh 86.1% / pps 70.7% / pal 51.9% / par 57.1% / mcp 63.6%）

| Pair | Task A | Δ | Task B | Δ |
|---|---|---|---|---|
| bbh + pps | 63.4% | −22.7 | 76.9% | +6.2 |
| bbh + pal | 55.7% | −30.4 | 48.2% | −3.7 |
| bbh + par | 76.5% | −9.6 | 59.2% | +2.1 |
| bbh + mcp | 39.9% | **−46.2**（全场最差）| 67.7% | +4.1 |
| pps + pal | 54.1% | −16.6 | 66.6% | +14.7 |
| pps + par | 70.7% | **0.0** | 70.5% | **+13.4** |
| pps + mcp | 72.0% | +1.3 | 46.4% | −17.2 |
| pal + par | 64.5% | **+12.6** | 56.0% | −1.1 |
| pal + mcp | 62.4% | +10.5 | 40.8% | −22.8 |
| par + mcp | 58.1% | +1.0 | 62.6% | −1.0 |

### 按任务平均掉分幅度（4 次配对均值）

| 任务 | 平均 Δ | 兼容性 |
|---|---|---|
| place_a2b_left (pal) | **+8.5pp** | 融合越多越好，兼容性最强 |
| place_a2b_right (par) | **+3.9pp** | 全场景兼容，"万能连接点" |
| place_phone_stand (pps) | −2.3pp | 基本中性，稳健 |
| move_can_pot (mcp) | −9.2pp | 中度脆弱，尤其怕 pal/pps |
| beat_block_hammer (bbh) | **−27.2pp** | 最脆弱，几乎和谁融合都大幅掉分 |

### 结论

- **推荐融合**：pps+par（0/+13.4，近乎无损）、pal+par（+12.6/−1.1）、par+mcp（+1.0/−1.0）——凡带 place_a2b_right 的组合都表现优异，是唯一的「全兼容」任务。
- **勉强可用**：bbh+par、pps+mcp、pal+mcp、pps+pal
- **不建议融合**：bbh+pps、bbh+pal，尤其 **bbh+mcp**（bbh 从 86.1% 暴跌至 39.9%，全场最差组合）。beat_block_hammer 单任务 PPO 最强，但与其余任务（除 par 外）融合都损失惨重，是 5 任务全量融合中 bbh 只剩 29-37% 的根本原因——bbh 与 pps、pal、mcp 均冲突，5 者叠加时冲突效应累积到极致。
- place_a2b_right 是结构上的「桥梁任务」：与它配对的组合全部不劣于任一单任务基线之和的期望；而 beat_block_hammer 是「冲突源」：其 PPO 更新方向与其他任务（par 除外）在共享 expert 层参数空间中方向相反或强烈重叠，导致相互抵消/破坏。
