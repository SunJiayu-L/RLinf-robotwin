# PPO 训练 Seeds 配置问题

**日期**：2026-06-26  
**影响任务**：click_alarmclock、move_playingcard_away、pick_dual_bottles

---

## 背景

RoboTwin 环境的 seeds 文件有两种格式，分别走不同的分区逻辑（`seed_utils.py`）：

| 格式 | 关键字段 | 分区函数 | 特性 |
|------|---------|---------|------|
| **success_seeds** | `{"task": {"success_seeds": [...]}}` | `partition_success_seeds` | 循环复用，**永不耗尽** |
| **episode manifest** | `{"task": {"episodes": [...]}}` | `partition_episode_manifest` | 线性消费，**严格整除性检查** |

判断逻辑在 `robotwin_env.py:_init_reset_state_ids`：先检查 `episodes` 字段，存在则走 manifest 路径；否则检查 `success_seeds` 字段。

---

## 错误一：Episode Manifest 运行时耗尽

### 错误信息

```
RuntimeError: Episode manifest exhausted before evaluation completed:
requested through 16, only 8 available
```

### 触发条件

训练 seeds 文件为 episode manifest 格式，且每 GPU 分配到的 episode 数 < 一个训练步需要消耗的 episode 数。

### 具体数值（click_alarmclock 原始配置）

```
seeds 文件：click_alarmclock_train64.json（64 条 episodes）
每 GPU 可用：64 / 8 GPU = 8 条
每训练步需要：rollout_epoch × (total_num_envs / num_gpus) = 4 × (64/8) = 32 条
```

第1轮 rollout 消耗 episodes [0..7]（共8条，OK）  
第2轮 rollout 请求 episodes [8..15]（"requested through 16"），只有 8 条 → **崩溃**

---

## 错误二：Episode Manifest 整除性检查失败

### 错误信息

```
ValueError: Each worker's episode count must be divisible by its number of
environment groups: 12 % 8 != 0
```

### 触发条件

`partition_episode_manifest` 有两层整除性检查：
1. `len(episodes) % total_num_processes == 0`（episodes 总数能被 GPU 数整除）
2. `episodes_per_worker % num_group == 0`（每 GPU 的 episodes 数能被该 GPU 的 env 组数整除）

### 具体数值（move_playingcard_away / pick_dual_bottles）

```
seeds 文件：manifest_96ep.json（96 条 episodes）
episodes_per_worker = 96 / 8 GPU = 12
num_group = total_num_envs / num_gpus = 64 / 8 = 8
检查：12 % 8 = 4 ≠ 0 → 初始化即崩溃
```

---

## 各任务原始配置对比

| Task | train seeds 文件 | 格式 | 每GPU条数 | 问题 |
|------|-----------------|------|----------|------|
| beat_block_hammer | `train_seeds.json` | success_seeds (1000条) | 循环 | ✅ 正常 |
| place_object_stand | `place_object_stand_train_seeds.json` | success_seeds (约1000条) | 循环 | ✅ 正常 |
| click_alarmclock | `click_alarmclock_train64.json` | **episodes (64条)** | 8条，需要32 | ❌ 错误一 |
| move_playingcard_away | `move_playingcard_away_manifest_96ep.json` | **episodes (96条)** | 12条，12%8≠0 | ❌ 错误二 |
| pick_dual_bottles | `pick_dual_bottles_manifest_96ep.json` | **episodes (96条)** | 12条，12%8≠0 | ❌ 错误二 |

---

## 修复方案

### 原则

训练 seeds 优先使用 **success_seeds 格式**（循环、无整除性约束）。仅当无对应文件时，用 `null`（随机初始化）。

### 各任务修复

| Task | 修复前 | 修复后 | 原因 |
|------|--------|--------|------|
| click_alarmclock | `click_alarmclock_train64.json`（episodes） | `click_alarmclock_train_seeds.json`（success_seeds, 64条） | 同任务有专用 success_seeds 文件 |
| pick_dual_bottles | `pick_dual_bottles_manifest_96ep.json`（episodes） | `train_seeds.json`（success_seeds, 1250条） | `train_seeds.json` 包含该任务的 key |
| move_playingcard_away | `move_playingcard_away_manifest_96ep.json`（episodes） | `seeds_path: null` | 无对应 success_seeds 文件，用随机初始化 |

### Config 修改位置

`examples/embodiment/config/moe_sft20k_{task}_ppo100.yaml` 中 `env.train.seeds_path`（第85行）。  
**eval seeds 不变**，eval 使用 `manifest_96ep.json`（32 eval envs / 8 GPU = 4 envs/GPU，96/8=12 eps/GPU，12%4=0 ✅）。

---

## 整除性约束公式

使用 episode manifest 作为训练 seeds 时，需满足：

```
条件1: total_seeds % num_gpus == 0
条件2: (total_seeds / num_gpus) % (total_num_train_envs / num_gpus) == 0
合并：  total_seeds % total_num_train_envs == 0
```

当前训练配置 `total_num_train_envs = 64`，因此 seeds 总数必须是 **64 的倍数**（64, 128, 192, 256, ...）。

此外还需保证 episodes 充足：
```
total_seeds / num_gpus >= rollout_epoch × (total_num_train_envs / num_gpus)
即 total_seeds >= rollout_epoch × total_num_train_envs = 4 × 64 = 256
```

综合：**最小安全 manifest 大小 = 256 条**（64 的倍数 且 ≥ 256）。

---

## 代码位置

| 文件 | 位置 | 说明 |
|------|------|------|
| `rlinf/envs/robotwin/seed_utils.py` | `partition_episode_manifest` | 整除性检查（行64, 71） |
| `rlinf/envs/robotwin/seed_utils.py` | `partition_success_seeds` | 循环分区，无整除性约束 |
| `rlinf/envs/robotwin/robotwin_env.py` | `_init_reset_state_ids`（行507） | 根据字段选择分区函数 |
| `rlinf/envs/robotwin/robotwin_env.py` | `_update_manifest_episodes`（行~625） | 运行时耗尽检查 |
| `rlinf/envs/robotwin/robotwin_env.py` | `update_reset_state_ids`（行551） | success_seeds 循环取模（行565） |

---

## 相关 Seeds 文件

| 文件 | 格式 | 任务 | 条数 |
|------|------|------|------|
| `train_seeds.json` | success_seeds | 22个任务（含 beat_block_hammer, pick_dual_bottles 等） | 1000~1250/任务 |
| `click_alarmclock_train_seeds.json` | success_seeds | click_alarmclock | 64 |
| `place_object_stand_train_seeds.json` | success_seeds | place_object_stand | ~1000 |
| `click_alarmclock_train64.json` | **episodes** | click_alarmclock | 64 |
| `{task}_manifest_96ep.json` | **episodes** | 各任务 | 96 |
