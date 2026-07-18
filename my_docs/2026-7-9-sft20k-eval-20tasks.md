# 2026-7-9 SFT-20k base 在 moe-demo-clean 20 任务上的完整评测

## 背景

`pi05_aloha_moe_demo_clean` 的 SFT-20k base checkpoint（`robotwin_moe_demo_clean_full_pytorch/20000`）此前从未在 `moe-demo-clean` 训练集覆盖的全部 20 个任务上系统评测过。本次目标：对每个任务跑一次标准 eval（`total_num_envs: 32`，`eval_rollout_epoch: 3`，`use_fixed_reset_state_ids: True`，96 集 eval manifest），拿到一份完整的 20 任务基线成功率表。

**moe-demo-clean 的真实 20 个任务**（按 `/mnt/petrelfs/muyao/jiayusun/moe-demo-clean/*-demo_clean-200` 目录核实，是本次唯一权威列表——过程中一度错误地对照过 `scripts/collect_robotwin_manifests.py` 里的 `TASKS_20`，那是另一个不相关的 legacy 列表，已修正）：

```
beat_block_hammer, click_alarmclock, click_bell, handover_block, move_can_pot,
move_playingcard_away, pick_diverse_bottles, pick_dual_bottles, place_a2b_left,
place_a2b_right, place_can_basket, place_cans_plasticbox, place_dual_shoes,
place_object_stand, place_phone_stand, put_bottles_dustbin, rotate_qrcode,
stack_blocks_two, stack_bowls_three, turn_switch
```

--- 

## 最终结果（20/20，success_once）

| 任务 | Success | | 任务 | Success |
|---|---|---|---|---|
| beat_block_hammer | 72.2% | | place_can_basket | 0.2% |
| click_alarmclock | 80.6% | | place_cans_plasticbox | 0% |
| click_bell | 39.6% | | place_dual_shoes | 0% |
| handover_block | 12.5% | | place_object_stand | 67.8% |
| move_can_pot | 52.7% | | place_phone_stand | 49.6% |
| move_playingcard_away | 67.6% | | put_bottles_dustbin | 0% |
| pick_diverse_bottles | 37.7% | | rotate_qrcode | 65.6% |
| pick_dual_bottles | 2.2% | | stack_blocks_two | 0% |
| place_a2b_left | 48.2% | | stack_bowls_three | 0% |
| place_a2b_right | 56.6% | | turn_switch | 37.8% |

- 平均 success_once ≈ 34.7%（20 任务简单平均，未加权）。
- 完全 0%：`place_cans_plasticbox`、`place_dual_shoes`、`put_bottles_dustbin`、`stack_blocks_two`、`stack_bowls_three`（5 个）。数值经原始日志核对（`array(0., dtype=float32)`），不是显示截断。
- 最高：`click_alarmclock` 80.6%；最低（非零）：`pick_dual_bottles` 2.2%。
- 这 5 个 0% 任务原因未深挖，后续如果要分析，方向是任务本身的难度/SFT 训练数据里这几个任务的样本量或质量。

---

## 过程中发现并修复的问题

评测过程本身暴露了好几个独立的、跟"eval 逻辑"无关的基础设施/配置 bug，记录下来避免以后重复踩坑。

### 1. eval seed manifest 覆盖率核查

一开始被告知"20 个任务除了 pick_diverse_bottles 都生成了 eval seed"，核查后发现：
- 对照 `TASKS_20`（错误的参照列表）时，误判又多缺 3 个（adjust_bottle / blocks_ranking_rgb / blocks_ranking_size）——这 3 个根本不在 moe-demo-clean 范围内，已撤销为它们误提交的 seed 生成 job。
- 对照 moe-demo-clean 真实 20 任务后确认：确实只缺 `pick_diverse_bottles`，其 eval seed 生成 job（10208195）当时仍在跑，跑了约 4.7 小时后 `COMPLETED`，生成 `pick_diverse_bottles_manifest_96ep.json`。

### 2. `click_alarmclock` 训练配置 seeds_path 指向不存在的文件

`moe_sft20k_click_alarmclock_ppo100.yaml` 的 `env.train.seeds_path` 写的是 `click_alarmclock_train_seeds.json`，该文件**不存在**。`_init_reset_state_ids()` 遇到文件不存在会静默退化为完全随机 seed（不报错，但精选的 1000 个 success seed 全部作废）。修复：改成跟 bbh/pal/mcp 一样直接指向共享的 `train_seeds.json`（1000 个 seed，验证充分）。

### 3. `place_object_stand` 训练：`total_num_envs` 128 导致 seed 分区归零崩溃

现象：8 张卡上必现 `RuntimeError: ZeroDivisionError`，出在 `rlinf/envs/robotwin/robotwin_env.py:588 update_reset_state_ids()`。

根因（比"seed 文件为空"更精确）：`place_object_stand_train_seeds.json` 有 64 个真实 seed，不是空文件。但 `seed_utils.py::partition_success_seeds()` 最后一步 `keep_count = (worker_seeds.numel() // num_group) * num_group` 会向下取整——`total_num_envs=128` 时每 worker 分到 8 个 seed，而 `num_group=16`，`8 // 16 = 0`，直接把该 worker 的 seed 截断成空。`total_num_envs=64` 时 `num_group=8 == 8` 刚好不归零。bbh/pal/mcp 等任务没事是因为它们用的共享 `train_seeds.json` 里每任务有 1000 个 seed，除 8 个 worker 后还有 125 个，远超 num_group=16。

修复：`moe_sft20k_place_object_stand_ppo100.yaml` 的 `env.train.total_num_envs` 从 128 改回 64（只改这一个任务，没碰代码）。验证：`partition_success_seeds(..., total_num_processes=8, num_group=8)` 每个 worker 返回 8 个非空 seed。

### 4. `click_bell`：核实 128 envs 安全，直接提交

1000 个 seed，`total_num_envs=128` 下每 worker 112 个，远超归零阈值，`seeds_path` 本身也已正确指向共享文件，无需改动即可用 128 提交。

### 5. 8 个任务缺失 `moe_sft20k_eval_<task>.yaml`

`place_can_basket / place_cans_plasticbox / place_dual_shoes / put_bottles_dustbin / rotate_qrcode / stack_blocks_two / stack_bowls_three / turn_switch` 8 个任务没有 SFT-20k eval 配置。照 `moe_sft20k_eval_beat_block_hammer.yaml` 模板用 `sed` 替换任务名批量生成。

### 6. 7 个任务的 `env/robotwin_<task>.yaml` 用了不兼容的 Hydra package 写法

生成上面 8 个 eval 配置后跑，除 `put_bottles_dustbin` 外全部秒崩：`omegaconf.errors.ConfigAttributeError: Missing key env_type`，`full_key: env.train.env_type`。

根因：`place_can_basket / place_cans_plasticbox / place_dual_shoes / rotate_qrcode / stack_blocks_two / stack_bowls_three / turn_switch` 这 7 个任务的 `examples/embodiment/config/env/robotwin_<task>.yaml` 文件头部是 `# @package _global_`，内容自带嵌套的 `env: {train: {...}, eval: {...}}` 结构；而 `moe_sft20k_eval_<task>.yaml` 用的是 `env/robotwin_<task>@env.train` / `@env.eval` 这种 package-override 写法，要求被引用文件是**扁平**结构（顶层直接 `env_type: ...`，不带 `env:` 包装，参见 `robotwin_beat_block_hammer.yaml`）。两种写法混用导致整个文件被错误地二次嵌套到 `env.train.env.train.*`，`env.train.env_type` 自然找不到。`put_bottles_dustbin` 本身用的就是扁平写法，所以没事。

修复：把这 7 个文件用 `robotwin_beat_block_hammer.yaml` 模板重写成扁平结构（task_name 替换），原文件备份为 `robotwin_<task>.yaml.bak_nested`。用 `hydra.compose()` 本地验证全部 7 个 `env.train.env_type` 正确解析后再提交。

### 7. 两次 job 因 worker 进程随机死亡而死等到超时

第一次（job 10208445，19 任务顺序跑）在 `handover_block` 跑到一半时，`Rollout:0` 的 `ChannelWorker` actor 意外死亡（`SYSTEM_ERROR`，connection error code 2 / EOF，可能 OOM 或 SIGSEGV），其余几个 `EnvGroup` rank 没有跟着崩，而是卡在等一个再也不会 ready 的 collective group 上，每隔 1.5 小时打一条"Waited N seconds"，一直空等到 24 小时 SLURM 时限（这个 job 本身被我们手动 `scancel` 掉，改为续跑剩余 16 个）。

第二次在续跑的 `place_object_stand` 上又遇到同一类偶发死亡（不同 worker port，同样的 `SYSTEM_ERROR`/EOF 特征），但这次因为脚本已经加了 `timeout 1800`（30 分钟）包住每个任务的 eval 命令，`pipefail` 下 `timeout` 的退出码 124 被正确捕获，自动标记 `[TIMEOUT]` 并跳到下一个任务，没有再拖垮整个 job。`place_object_stand` 单独重试后一次性通过（67.8%）。

**结论**：这个 worker 死亡看起来是偶发性的基础设施问题（不是特定任务导致），暂未深挖具体触发条件；实际防护手段是给每个 eval/训练任务加超时熔断，避免死等吃满时限。

---

## 涉及的文件改动

| 文件 | 改动 |
|---|---|
| `examples/embodiment/config/moe_sft20k_click_alarmclock_ppo100.yaml` | `env.train.seeds_path` 改指向共享 `train_seeds.json` |
| `examples/embodiment/config/moe_sft20k_click_bell_ppo100.yaml` | `env.train.total_num_envs` 64→128 |
| `examples/embodiment/config/moe_sft20k_place_object_stand_ppo100.yaml` | `env.train.total_num_envs` 128→64（修 ZeroDivisionError） |
| `examples/embodiment/config/moe_sft20k_eval_{place_can_basket,place_cans_plasticbox,place_dual_shoes,put_bottles_dustbin,rotate_qrcode,stack_blocks_two,stack_bowls_three,turn_switch,pick_diverse_bottles}.yaml` | 新建（9 个，模板化生成） |
| `examples/embodiment/config/env/robotwin_{place_can_basket,place_cans_plasticbox,place_dual_shoes,rotate_qrcode,stack_blocks_two,stack_bowls_three,turn_switch,pick_diverse_bottles}.yaml` | 重写为扁平结构（8 个，原文件备份为 `.bak_nested`） |
| `examples/embodiment/slurm/moe_sft20k_eval_all19.sbatch` | 新建，19 任务顺序 eval（第一轮，job 10208445，中途死锁被 scancel） |
| `examples/embodiment/slurm/moe_sft20k_eval_remaining16.sbatch` | 新建，续跑剩余 16 个 + 30 分钟 timeout 熔断（job 10210142） |
| `examples/embodiment/slurm/moe_sft20k_eval_retry8.sbatch` | 新建，重试 7 个配置修复任务 + place_object_stand（job 10210921） |
| `examples/embodiment/slurm/moe_sft20k_eval_pick_diverse_bottles.sbatch` | 新建，单任务 eval（job 10211234） |

**SLURM job 记录**：10208195（pick_diverse_bottles 生成 eval seed）、10208323（click_alarmclock 训练重提）、10208332（click_bell 训练重提）、10207370（place_object_stand 训练重提，验证 seed 修复生效）、10208445→scancel、10210142、10210921、10211234（本次 eval 主线）。
