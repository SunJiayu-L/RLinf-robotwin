# PPO 训练全量记录

**基础模型：** `pi05_aloha_moe_demo_clean`，SFT-20k checkpoint（`robotwin_moe_demo_clean_full_pytorch/20000`）  
**框架：** RLinf · 8 GPU（A100）· JAX flow-matching VLA  
**最后更新：** 2026-07-17

---

## SFT-20k 基线（标准协议）

使用 `eval_rollout_epoch: 3`，96ep manifest，`max_episode_steps: 400`。完整记录见 `my_docs/2026-7-9-sft20k-eval-20tasks.md`。

| 任务 | SFT-20k 基线 |   | 任务 | SFT-20k 基线 |
|---|---|---|---|---|
| beat_block_hammer | **72.2%** | | place_object_stand | **67.8%** |
| click_alarmclock | **80.6%** | | place_phone_stand | **49.6%** |
| click_bell | **39.6%** | | put_bottles_dustbin | **0%** |
| handover_block | **12.5%** | | rotate_qrcode | **65.6%** |
| move_can_pot | **52.7%** | | turn_switch | **37.8%** |
| move_playingcard_away | **67.6%** | | place_a2b_left | **48.2%** |
| pick_diverse_bottles | **37.7%** | | place_a2b_right | **56.6%** |

> 注：部分单任务 PPO 实验使用较早的基线（评测协议略有差异），已在各节中标注。

---

## 一、单任务 PPO

所有单任务运行使用 **8 GPU（1 节点）**，`global_batch_size: 2048`，`max_episode_steps: 400`。

### 任务一览

| 简称 | 任务名 | Log 目录（日期前缀） | 当前位置 | Checkpoints | 大小 | SFT 基线 | 最优 PPO eval |
|---|---|---|---|---|---|---|---|
| BBH-v1 | beat_block_hammer | `20260625-11:37:02` | nvme/logs | step100/200/300 | — | 57.8%¹ | **86.1%** @step299 ⭐ |
| BBH-v2 | beat_block_hammer | `20260707-22:22:21` | petrelfs | step50/100 | 38G | 72.2% | — |
| CA | click_alarmclock | `20260708-13:50:33` | petrelfs | step50 | 19G | 80.6% | — |
| CB | click_bell | `20260708-13:56:28` | petrelfs | step50/100 | 38G | 39.6% | — |
| MPA | move_playingcard_away | `20260706-11:28:57` | petrelfs | step50/100/150 | 57G | 67.6% | — |
| PAR | place_a2b_right | `20260705-10:45:17` | petrelfs | step50/200² | 38G | 56.6% | **57.1%** @step49 |
| PAL | place_a2b_left | `20260706-23:28:22` | petrelfs | step50/100 | 38G | 48.2% | **51.9%** @step299 |
| PPS-v1 | place_phone_stand | `20260710-02:54:19` | petrelfs | step50/100 | 38G | 49.6% | **70.7%** @step299 |
| PPS-v2 | place_phone_stand | `20260716-21:41:25` | nvme/logs_july17 | step50 | 23G | 49.6% | — |
| POS | place_object_stand | `20260710-02:54:16` | petrelfs | step50/100 | 38G | 67.8% | **65.9%** @step299³ |
| PDB | pick_diverse_bottles | `20260710-02:54:19` | petrelfs | step50/100 | 38G | 37.7% | — |
| MCP | move_can_pot | `20260706-23:28:23` | petrelfs | step50/100 | 38G | 52.7% | **63.6%** @step399 |
| PBD | put_bottles_dustbin | `20260705-21:24:24` | petrelfs | step50 | 19G | 0% | 放弃（全零奖励） |
| TS | turn_switch | `20260716-21:39:48` | nvme/logs_july17 | step50/100 | 39G | 37.8% | — |
| RQ | rotate_qrcode | `20260716-21:41:25` | nvme/logs_july17 | step50 | ~30G | 65.6% | — |
| HB-fail | handover_block | 多次失败（v1/v2）| nvme/logs | — | — | 12.5% | 放弃（早期方案） |

> ¹ BBH-v1 使用旧评测协议，SFT baseline 为 57.8%，新协议为 72.2%。  
> ² PAR step200 为手动暂停后续跑，非 step100 save interval 正常产出。  
> ³ POS PPO eval 低于 SFT 基线，PPO 无明显提升。

---

### 重要单任务详细记录

#### beat_block_hammer PPO v1（最优结果）

- **Log：** `logs/20260625-11:37:02-moe_sft20k_beat_block_hammer_ppo100/`（已迁移到 nvme/logs）
- **配置：** 标准 v1，128 envs
- **结果：** SFT 57.8% → PPO eval **86.1%** @step299（**+28.3pp** ⭐ 全任务最高涨点）
- **状态：** job 超时，中止于 step 335，step300 为最优点
- **Checkpoint 路径：** `/nvme/jiayusun/data_backup/logs/<dir>/checkpoints/global_step_300/`

#### move_playingcard_away PPO（最多 steps）

- **Log：** `logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100/`（petrelfs）
- **配置：** 64 envs（旧），step50/100/150 均存
- **大小：** 57G（3 个 checkpoint）
- **SFT 基线：** 67.6%（新协议）

#### place_phone_stand PPO（最高单次 eval）

- **Log：** `logs/20260710-02:54:19-moe_sft20k_place_phone_stand_ppo100/`（petrelfs）
- **配置：** 128 envs，标准 v1
- **结果：** SFT 49.6% → PPO eval **70.7%** @step299（+21.1pp）
- **备注：** step249 eval 仅 33.3%（大方差），step299 跳升至 70.7%

#### put_bottles_dustbin PPO（放弃）

- SFT 对此任务 success 率 0%，全零奖励，PPO 无法收敛，step96 后 scancel。

---

## 二、handover_block 专项

### HB-v1/v2（早期方案，均失败）

- **v1（Job 10192376）：** 稀疏奖励（init train success ~9%），kl_beta=0，lr=5e-6，PPO 训练后 29.2% < SFT init eval 37.5%，放弃。
- **v2（Job 10200331/10201021）：** filter_rewards bug 导致全过滤；修复后 collective 通信异常崩溃，放弃。
- **根因：** handover_block 属稀疏奖励任务，标准 v1 超参（kl_beta=0, lr=5e-6）不适用。

### HB-128env ppo150

- **Log：** `logs/20260715-05:51:55-moe_sft20k_handover_block_ppo150_128env/`（已迁移至 nvme/logs_july17）
- **配置：** 8 GPU，128 envs，v2 超参（lr=2e-6, kl_beta=0.02, rollout_epoch=8）
- **Checkpoint：** step150（step50/100 已删，冗余）
- **用途：** step150 作为 ppo150to500 续训起点

### HB-128env ppo150to500（进行中）

- **Log：** `logs/20260716-19:02:25-moe_sft20k_handover_block_ppo150to500_128env/`（**petrelfs**，保留）
- **配置：** 8 GPU，128 envs，从 step150 checkpoint resume
- **当前：** step200/250 checkpoint，约 20G
- **Checkpoint 路径：**
  ```
  logs/20260716-19:02:25-.../checkpoints/global_step_200/
  logs/20260716-19:02:25-.../checkpoints/global_step_250/
  ```

### HB-KL 约束（开发中，Branch: KL）

- 在 handover_block 单任务 PPO loss 中加入对 5-task 联合策略的 KL 约束：
  `L = L_PPO + kl_beta * D_KL(π_hb || π_5task_joint)`
- **参考策略：** 5-task joint PPO step100 checkpoint
- **改动文件：** `rlinf/algorithms/losses.py`、`rlinf/algorithms/utils.py`、`rlinf/workers/actor/fsdp_actor_worker.py`
- 详见 `my_docs/2026-07-18-handover-block-ppo-kl-constraint.md`

---

## 三、多任务联合 PPO

### 5-task（BBH + CA + CB + HB + MPA）— 主要实验

| 字段 | 值 |
|---|---|
| **Log 目录** | `logs/20260712-13:41:23-moe_ppo_5task_resume50to300_bbh_ca_cb_hb_mpa/` |
| **当前位置** | petrelfs |
| **GPU 配置** | 原始训练：24 GPU（3 节点 × 8 GPU）；resume：8 GPU（1 节点） |
| **Train envs** | 128（每节点），Eval envs：40 |
| **训练范围** | step50（24GPU run 产出）→ step200（resume run 继续） |
| **Checkpoints** | step100 / step150 / step200 |
| **大小** | 57G |
| **SFT baselines** | BBH 72.2%，CA 80.6%，CB 39.6%，HB 12.5%，MPA 67.6% |
| **Sbatch** | `moe_ppo_5task_24gpu_bbh_ca_cb_hb_mpa.sbatch`（原始）<br>`moe_ppo_5task_resume50to300_bbh_ca_cb_hb_mpa.sbatch`（resume）|

**5-task 在 5 个任务上的 eval（step50 checkpoint，来自 `results/5ppo_on_5task/`）：**

| | BBH | CA | CB | MCP | MPA | 平均 |
|---|---|---|---|---|---|---|
| **5task joint step50** | ~72% | ~100% | ~99% | — | ~90% | — |

> 注：此结果来自将 5-task 权重（step50 eval manifest）在各任务分别评测，MCP 未完成。

**Checkpoint 路径（petrelfs）：**
```
logs/20260712-13:41:23-moe_ppo_5task_resume50to300_bbh_ca_cb_hb_mpa/
  moe_ppo_5task_resume50to300_bbh_ca_cb_hb_mpa/checkpoints/
    global_step_100/actor/model_state_dict/full_weights.pt  ← KL 约束实验用此权重
    global_step_150/
    global_step_200/
```

---

### 5-task（BBH + CA + CB + MPA + MCP）

| 字段 | 值 |
|---|---|
| **Log 目录** | `logs/20260715-12:29:28-moe_ppo_5task_bbh_ca_cb_mpa_mcp/` |
| **当前位置** | nvme/logs_july17（已迁移） |
| **GPU 配置** | 8 GPU（1 节点） |
| **Checkpoints** | step50 / step100 |
| **大小** | 39G |
| **Sbatch** | `moe_ppo_5task_bbh_ca_cb_mpa_mcp.sbatch` |
| **与上一个 5-task 区别** | 无 handover_block，新增 move_can_pot |

---

### 4-task（PAL + PAR + MCP + PPS）

| 字段 | 值 |
|---|---|
| **Log 目录** | `logs/20260715-17:21:21-moe_ppo_4task_pal_par_mcp_pps/` |
| **当前位置** | nvme/logs_july17（已迁移） |
| **GPU 配置** | 8 GPU（1 节点） |
| **Checkpoints** | step50 / step100 |
| **大小** | 39G |
| **Sbatch** | `moe_ppo_4task_pal_par_mcp_pps.sbatch` |

---

## 四、Checkpoint 位置速查

### petrelfs（当前可直接使用）

```
logs/
├── 20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100/          38G  step50/200
├── 20260705-21:24:24-moe_sft20k_put_bottles_dustbin_ppo100/      19G  step50
├── 20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100/    57G  step50/100/150
├── 20260706-23:28:22-moe_sft20k_place_a2b_left_ppo100/           38G  step50/100
├── 20260706-23:28:23-moe_sft20k_move_can_pot_ppo100/             38G  step50/100
├── 20260707-22:22:21-moe_sft20k_beat_block_hammer_ppo100/        38G  step50/100
├── 20260708-13:50:33-moe_sft20k_click_alarmclock_ppo100/         19G  step50
├── 20260708-13:56:28-moe_sft20k_click_bell_ppo100/               38G  step50/100
├── 20260710-02:54:16-moe_sft20k_place_object_stand_ppo100/       38G  step50/100
├── 20260710-02:54:19-moe_sft20k_pick_diverse_bottles_ppo100/     38G  step50/100
├── 20260710-02:54:19-moe_sft20k_place_phone_stand_ppo100/        38G  step50/100
├── 20260712-13:41:23-moe_ppo_5task_resume50to300_bbh_ca_cb_hb_mpa/  57G  step100/150/200
└── 20260716-19:02:25-moe_sft20k_handover_block_ppo150to500_128env/  20G  step200/250
```

### node-10 /nvme/jiayusun/data_backup/

| 路径 | 内容 |
|---|---|
| `logs/` | 2026-07-15 迁移的 425 个旧目录，含 BBH-v1 最优权重（step300，86.1%）|
| `logs_july17/20260715-05:51:55-moe_sft20k_handover_block_ppo150_128env/` | HB step150 |
| `logs_july17/20260715-12:29:28-moe_ppo_5task_bbh_ca_cb_mpa_mcp/` | 5task-v2 step50/100 |
| `logs_july17/20260715-17:21:21-moe_ppo_4task_pal_par_mcp_pps/` | 4task step50/100 |
| `logs_july17/20260716-21:39:48-moe_sft20k_turn_switch_ppo100/` | TS step50/100 |
| `logs_july17/20260716-21:41:25-moe_sft20k_rotate_qrcode_ppo100/` | RQ step50 |
| `logs_july17/20260716-21:41:25-moe_sft20k_place_phone_stand_ppo100/` | PPS-v2 step50 |

从 node-10 恢复：`SLURM job --nodelist=SH-IDCA1404-10-140-54-10`，rsync 目标目录回 petrelfs logs/。

---

## 五、超参对照

### 标准 v1（大多数单任务）

```yaml
actor:
  micro_batch_size: 32 (或 64)
  global_batch_size: 2048
  optim:
    lr: 5.0e-06
algorithm:
  rollout_epoch: 4 (或 8)
  kl_beta: 0.0
  filter_rewards: False
  update_epoch: 5
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
env:
  train:
    total_num_envs: 128
    max_steps_per_rollout_epoch: 400
  eval:
    max_episode_steps: 400
val_check_interval: 50
save_interval: 50
```

### v2（稀疏奖励 / handover_block）

```yaml
actor:
  micro_batch_size: 64
  global_batch_size: 2048
  optim:
    lr: 2.0e-06
algorithm:
  rollout_epoch: 8
  kl_beta: 0.02
  filter_rewards: False
```

---

## 六、关键经验

1. **最高涨点：** BBH-v1 SFT→PPO +28.3pp（86.1%），PPS +21.1pp（70.7%），PAL +16.5pp（51.9%）。
2. **无明显提升：** POS（SFT 67.8% vs PPO 65.9%），PAR 仅 +2pp 且 SFT 已高。
3. **PPO 有害：** HB 早期方案 PPO 后低于 init eval，稀疏任务需专用超参。
4. **全零奖励放弃：** PBD（SFT 0%），PPO 无法启动。
5. **eval 方差：** 32 envs × 1 epoch = 32 eps，±10pp 波动正常；训练后务必用 `eval_rollout_epoch: 3` 取平均。
6. **eval step = 400：** 200-step eval 系统低估 success rate（click_alarmclock 消融验证）。
7. **稀疏任务 filter_rewards：** chunk-level reward ~0.0002，若 `rewards_lower_bound: 0.1` 会静默过滤全部数据（零报错）。
