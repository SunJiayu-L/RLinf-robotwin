# /nvme 数据迁移记录

**日期：** 2026-07-14  
**原因：** petrelfs 配额接近上限（~15.97 / 16.15 TiB），需腾出约 500 GB 空间用于 `robotwin_moe_demo_clean_hb1to1` 数据集转换（~424 GB）及后续 SFT 训练检查点（~210 GB）。  
**方法：** rsync 到计算节点 /nvme 本地存储（不在 job 结束后清理，验证持久化有效），再删除 petrelfs 副本。

---

## 迁移节点

| 节点 | /nvme 总容量 | 迁移前可用 | 迁移后可用（预估）|
|---|---|---|---|
| SH-IDCA1404-10-140-54-10 | 3.5 TB | 2.3 TB | ~1.8 TB |

> 注：该节点在训练/eval sbatch 的 `--exclude` 列表中（GPU 相关问题），但 CPU-only rsync job 可通过 `--nodelist` 直接访问。

**迁移 job：** SLURM job 10257545，耗时约 26 分钟（~310 MB/s）

---

## 迁移内容

### 1. LeRobot 数据集 — `robotwin_moe_demo_clean`
| 项目 | 路径 |
|---|---|
| **petrelfs 原路径（已删除）** | `/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_moe_demo_clean/` |
| **/nvme 备份路径** | `/nvme/jiayusun/data_backup/lerobot_robotwin_moe_demo_clean/` |
| 大小 | ~218 GB |
| 内容 | 4000 个 episode（20 个任务 × 200 episode），parquet 格式 |
| 备注 | 可通过 `moe-demo-clean/`（raw data，88 GB，**未迁移**）重新转换恢复 |

### 2. Pi0.5 SFT 中间检查点 — steps 5000 / 10000 / 15000
| 项目 | 路径 |
|---|---|
| **petrelfs 原路径（已删除）** | `/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full/{5000,10000,15000}/` |
| **/nvme 备份路径** | `/nvme/jiayusun/data_backup/pi05_ckpt_intermediate/{5000,10000,15000}/` |
| 大小 | ~42 GB × 3 = ~126 GB |
| 内容 | JAX 格式，pi05_aloha_moe_demo_clean 训练中间点 |
| 备注 | **step 20000（最终检查点）保留在 petrelfs**，路径：`.../robotwin_moe_demo_clean_full/20000/`，SFT 训练必需 |

### 3. 合并权重 — `merged_weights`
| 项目 | 路径 |
|---|---|
| **petrelfs 原路径（已删除）** | `/mnt/petrelfs/muyao/jiayusun/merged_weights/` |
| **/nvme 备份路径** | `/nvme/jiayusun/data_backup/merged_weights/` |
| 大小 | ~157 GB |
| 内容 | 23 个合并权重目录（各 ~6.8-8 GB），包括各种 MoE 实验组合 |

---

## 空间效果

| | 之前 | 之后 |
|---|---|---|
| 释放空间 | — | +501 GB |
| petrelfs 可用（估算） | ~184 GB | ~685 GB |
| 新数据集需求（7600 eps） | — | ~424 GB |
| SFT 训练检查点需求（5 步） | — | ~210 GB |
| 预计剩余 buffer | — | ~51 GB |

---

## 恢复方法

数据全部存活在 node-10 的 /nvme，SFT 训练完成后通过以下 sbatch 恢复：

```bash
sbatch examples/embodiment/slurm/move_data_back_from_nvme_node10.sbatch
```

该脚本将三份数据 rsync 回原始 petrelfs 路径。

> **注意：** /nvme 数据无自动清理，但节点存储为共享资源。如果 node-10 的 /nvme 被其他用户写满，数据不会丢失（只读不清理），但建议在 SFT 训练完成后尽快恢复。

---

## 时间线

| 时间 | 事件 |
|---|---|
| 2026-07-14 22:43 | job 10257545 提交，node-10 开始 rsync |
| 2026-07-14 23:09 | rsync 完成，验证通过（4000 parquet，23 merged dirs） |
| 2026-07-14 23:09 | 删除 petrelfs 三份数据，释放 501 GB |
| 2026-07-14 23:10 | 提交 convert job 10257573（hb1to1 数据集转换） |
| 2026-07-15 ~11:00 | convert job 10257584 OOM killed（exit 9），在 4601/7600 episode 处中断，`--mem=128G` 不足 |
| 2026-07-15 13:09 | 提交 convert 续传 job 10259265（新脚本 `convert_aloha_data_to_lerobot_robotwin_resume.py`，`--mem=256G`，从第 4601 episode 继续） |
| 2026-07-15 13:10 | 提交 log 备份 job 10259282（425 个旧 log 目录 → `/nvme/jiayusun/data_backup/logs/`，285 GB） |
| 2026-07-15 13:37 | log 备份完成（425 copied, 0 errors，node-10 /nvme 使用 2.1T/3.5T） |
| 2026-07-15 ~13:38 | convert job 10259265 因 EDQUOT（disk quota exceeded）失败——备份已完成但旧 log 尚未删除，quota 耗尽 |
| 2026-07-15 14:24 | 删除 job 10259514 完成：425 个目录从 petrelfs 删除，logs/ 从 740G → 455G，quota 从 16.01 → 15.66 TiB（释放 ~360 GB） |
| 2026-07-15 14:24 | 取消 job 10259265，提交新 convert 续传 job 10259517（同样从 4601 episode 续） |
| 2026-07-15 14:29 | 提交 job 10259603，将 `robotwin_moe_demo_clean`（218G，4000 parquets）rsync 到 node-10 /nvme |
| 2026-07-15 14:42 | rsync 完成，验证通过（4000 parquets 匹配），job 10259605 删除 petrelfs 副本 |
| 2026-07-15 14:42 | 删除完成，配额 15.71 TiB（convert job 同时写入约 260G 新数据，净余 ~440 GB） |

---

## 第二次迁移 — 旧 PPO log 目录（2026-07-15）

**原因：** convert 续传 job 因 EDQUOT 失败，需先释放 petrelfs 空间。  
**方法：** rsync 425 个旧/重复 log 目录到 node-10 /nvme，验证后删除。

### 迁移内容

| 项目 | 路径 |
|---|---|
| **petrelfs 原路径（已删除）** | `/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/<425 个旧目录>` |
| **/nvme 备份路径** | `/nvme/jiayusun/data_backup/logs/` |
| 大小 | ~285 GB |
| 内容 | June 2026 eval 实验、7月重复/失败的 PPO 训练 log（保留每个 task 最新的 best run） |

### 保留在 petrelfs 的重要 log（不迁移）

| 目录 | 内容 |
|---|---|
| `20260712-13:41:23-moe_ppo_5task_resume50to300_bbh_ca_cb_hb_mpa` | 5-task best（57G，step 50→300） |
| `20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100` | MPA single-task best（57G） |
| `20260710-02:54:19-moe_sft20k_place_phone_stand_ppo100` | PPS best（38G） |
| `20260710-02:54:19-moe_sft20k_pick_diverse_bottles_ppo100` | PDB best（38G） |
| `20260710-02:54:16-moe_sft20k_place_object_stand_ppo100` | POS best（38G） |
| `20260708-13:56:28-moe_sft20k_click_bell_ppo100` | CB best（38G） |
| `20260707-22:22:21-moe_sft20k_beat_block_hammer_ppo100` | BBH best（38G） |
| `20260706-23:28:23-moe_sft20k_move_can_pot_ppo100` | MCP best（38G） |
| `20260706-23:28:22-moe_sft20k_place_a2b_left_ppo100` | PAL best（38G） |
| `20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100` | PAR best（38G） |
| `20260705-21:24:24-moe_sft20k_put_bottles_dustbin_ppo100` | PBD best（19G） |
| `20260708-13:50:33-moe_sft20k_click_alarmclock_ppo100` | CA best（19G） |
| 当前运行中的 log 目录 | hb-ppo150、5task-mcp 等 |

### 恢复方法

```bash
# 从 node-10 /nvme 恢复旧 log 到 petrelfs（如需要）：
# SLURM job with --nodelist=SH-IDCA1404-10-140-54-10
rsync -a /nvme/jiayusun/data_backup/logs/ \
    /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/
```

---

## 第三次迁移 — 2026-07-17 PPO log 目录

**原因：** rq/pps step100 checkpoint 写入失败（EDQUOT），quota 再次打满。  
**方法：** SLURM job 10266156，nodelist=node-10，rsync → `/nvme/jiayusun/data_backup/logs_july17/`，验证后删除。  
**sbatch：** `examples/embodiment/slurm/migrate_logs_july17_to_node10.sbatch`

### 迁移前立即删除（直接 rm，不备份）

| 项目 | 原因 |
|---|---|
| `rq.../checkpoints/global_step_100/` (4.7G) | 写入时报错，文件损坏 |
| `20260715-17:21:21-moe_sft20k_place_phone_stand_ppo100/` (2.4M) | 失败启动，近空 |
| `20260715-17:21:21-moe_sft20k_place_object_stand_ppo100/` (2.5M) | 失败启动，近空 |
| `hb-ppo150.../checkpoints/global_step_50/` (~19G) | step150 已用于 resume ppo500，step50/100 冗余 |
| `hb-ppo150.../checkpoints/global_step_100/` (~19G) | 同上 |

### 迁移到 node-10 /nvme（job 10266156）

目标路径：`/nvme/jiayusun/data_backup/logs_july17/`

| 目录 | 大小（迁移时） | 说明 |
|---|---|---|
| `20260715-05:51:55-moe_sft20k_handover_block_ppo150_128env` | ~20G | 仅剩 step150 ckpt |
| `20260715-12:29:28-moe_ppo_5task_bbh_ca_cb_mpa_mcp` | 39G | step50/100 ckpts，eval 完成 |
| `20260715-17:21:21-moe_ppo_4task_pal_par_mcp_pps` | 39G | 已完成 |
| `20260716-21:39:48-moe_sft20k_turn_switch_ppo100` | 39G | step50/100 ckpts |
| `20260716-21:41:25-moe_sft20k_rotate_qrcode_ppo100` | ~30G | step50 ckpt（step100 已删） |
| `20260716-21:41:25-moe_sft20k_place_phone_stand_ppo100` | 23G | step50 ckpt（step100 未保存） |

预计释放：~190G

### 保留在 petrelfs（不迁移）

| 目录 | 原因 |
|---|---|
| `20260716-19:02:25-moe_sft20k_handover_block_ppo150to500_128env` (20G) | step200/250 ckpt，可能继续训练 |

### 时间线

| 时间 | 事件 |
|---|---|
| 2026-07-17 | rq/pps checkpoint 写入失败，确认 EDQUOT |
| 2026-07-17 | 直接删除冗余文件释放 ~43G |
| 2026-07-17 | 提交 job 10266156，迁移 ~190G 到 node-10 `/nvme/jiayusun/data_backup/logs_july17/` |

### 恢复方法（logs_july17）

```bash
# SLURM job with --nodelist=SH-IDCA1404-10-140-54-10
rsync -a /nvme/jiayusun/data_backup/logs_july17/ \
    /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/
```
