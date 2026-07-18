# moe-demo-clean SFT + PPO 实验记录

**日期**：2026-06-24  
**模型**：PI05（`pi05_aloha_moe_demo_clean`）  
**平台**：RLinf + RoboTwin-RLinf  
**机械臂**：aloha-agilex

---

## 1. 数据集：moe-demo-clean

### 1.1 基本信息

- **原始路径**：`/mnt/petrelfs/muyao/jiayusun/moe-demo-clean`
- **规模**：20 个任务，每任务 200 episodes，共 4000 episodes
- **LeRobot 数据集 ID**：`robotwin_moe_demo_clean`
- **缓存路径**：`/mnt/petrelfs/muyao/jiayusun/store/cache`

### 1.2 转换

- **sbatch**：`examples/embodiment/slurm/convert_moe_demo_clean_to_lerobot.sbatch`
- **转换 Job**：9923312，COMPLETED（7h01m）
- **输出**：4000/4000 episodes，约 813k frames

---

## 2. SFT 训练

### 2.1 配置

| 项目 | 值 |
|------|-----|
| OpenPI Config | `pi05_aloha_moe_demo_clean` |
| RLinf Dataconfig | `pi05_aloha_moe_demo_clean` |
| 初始权重 | `gs://openpi-assets/checkpoints/pi05_base/params`（JAX） |
| 总步数 | 20,000 |
| batch_size | 64 |
| GPU 数 | 4 |
| save_interval | 5000 |
| sbatch | `examples/embodiment/slurm/finetune_moe_demo_clean_pi05.sbatch` |
| experiment_name | `robotwin_moe_demo_clean_full` |

### 2.2 OpenPI Config 新增（两处）

**1. pi05 训练配置**（`/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/src/openpi/training/config.py`）：

```python
TrainConfig(
    name="pi05_aloha_moe_demo_clean",
    model=pi0_config.Pi0Config(pi05=True),
    data=LeRobotAlohaDataConfig(
        repo_id="robotwin_moe_demo_clean",
        adapt_to_pi=False,
        ...
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=20_000,
    batch_size=64,
    save_interval=5000,
    fsdp_devices=1,
)
```

**2. RLinf Dataconfig**（`rlinf/models/embodiment/openpi/dataconfig/__init__.py`）：

```python
TrainConfig(
    name="pi05_aloha_moe_demo_clean",
    model=pi0_config.Pi0Config(pi05=True, discrete_state_input=True),
    data=LeRobotAlohaDataConfig(
        repo_id="robotwin_moe_demo_clean",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(assets_dir="assets"),
        adapt_to_pi=False,
        extra_delta_transform=True,
    ),
    pytorch_weight_path="checkpoints/torch/pi05_base",
    num_train_steps=20_000,
)
```

### 2.3 训练记录

| Job ID | 状态 | 说明 |
|--------|------|------|
| 10081222 | COMPLETED ✅ | node-37，训练至 step 20000，耗时约 8.3h |

**Checkpoint 路径**：
```
/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full/
├── 5000/
├── 10000/
├── 15000/
└── 20000/
```

---

## 3. JAX → PyTorch 转换

转换脚本：`openpi/examples/convert_jax_model_to_pytorch.py`

| Checkpoint | Job ID | 状态 | 输出路径 |
|-----------|--------|------|---------|
| 5k | 10128806 | COMPLETED ✅ | `.../robotwin_moe_demo_clean_full_pytorch/5000/` |
| 20k | 10135950 | COMPLETED ✅ | `.../robotwin_moe_demo_clean_full_pytorch/20000/` |

**Pytorch 权重根路径**：
```
/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/
```

### 3.1 Norm Stats 路径问题与修复

**问题**：RLinf 的 `_override_with_model_path` 会覆盖 `assets_dir`，导致 norm_stats 查找路径为
`{model_path}/robotwin_moe_demo_clean/norm_stats.json`（缺少 `assets/` 前缀）。

**实际路径**：`{model_path}/assets/robotwin_moe_demo_clean/norm_stats.json`

**修复**：在每个 pytorch checkpoint 目录下创建 symlink：
```bash
ln -s {model_path}/assets/robotwin_moe_demo_clean {model_path}/robotwin_moe_demo_clean
```

20k 转换 sbatch 已内置该 symlink 创建步骤。

---

## 4. Eval 配置说明

- **eval 环境**：clean 场景（`random_background: false, cluttered_table: false, clean_background_rate: 1`）
- **max_episode_steps**：400
- **eval seeds**：`{task}_manifest_96ep.json`（96 episodes per task）
- **4 GPU eval** config 命名：`moe_sft5k_eval_{task}`
- **8 GPU eval** config 命名：`moe_sft20k_eval_{task}`

---

## 5. SFT 5k Eval 结果（4 GPU，96 episodes/task）

| Task | success_once |
|------|-------------|
| click_alarmclock | **86.7%** |
| click_bell | **65.6%** |
| beat_block_hammer | **52.3%** |
| move_playingcard_away | **39.0%** |
| place_object_stand | **34.5%** |
| place_a2b_right | **22.1%** |
| place_a2b_left | **13.9%** |
| move_can_pot | **10.8%** |
| pick_dual_bottles | **9.2%** |
| handover_block | **2.1%** |
| place_phone_stand | — |

---

## 6. SFT 20k Eval 结果（8 GPU，96 episodes/task）

| Task | SFT 5k | SFT 20k | 变化 |
|------|--------|---------|------|
| move_playingcard_away | 39.0% | **72.3%** | +33.3 pp |
| place_object_stand | 34.5% | **68.6%** | +34.1 pp |
| click_alarmclock | 86.7% | **67.3%** | -19.4 pp |
| place_a2b_right | 22.1% | **55.1%** | +33.0 pp |
| beat_block_hammer | 52.3% | **57.8%** | +5.5 pp |
| move_can_pot | 10.8% | **52.6%** | +41.8 pp |
| place_phone_stand | — | **39.6%** | — |
| place_a2b_left | 13.9% | **35.4%** | +21.5 pp |
| click_bell | 65.6% | **32.3%** | -33.3 pp |
| handover_block | 2.1% | **21.9%** | +19.8 pp |
| pick_dual_bottles | 9.2% | — (pending) | — |

**注**：click_alarmclock 和 click_bell 从 5k 到 20k 明显下降，可能 5k 时恰好
在这两个任务上过拟合较好，或 20k 在其他任务上的训练导致这两个任务性能退化。

---

## 7. PPO 实验（从 SFT 20k 初始化，500 steps）

### 7.1 配置

| 项目 | 值 |
|------|-----|
| 初始权重 | SFT 20k pytorch checkpoint |
| max_steps | 500 |
| GPU | 8 |
| total_num_envs（train） | 64 |
| total_num_envs（eval） | 32 |
| max_episode_steps | 400 |
| train seeds | `train_seeds.json` |
| eval seeds | `{task}_manifest_96ep.json` |
| val_check_interval | 50 |
| save_interval | 50 |
| Config 前缀 | `moe_sft20k_{task}_ppo100` |

详细记录见：`my_docs/2026-7-5-ppo-sft20k-experiment-log.md`

### 7.2 结果汇总（截至 2026-07-05）

| Task | SFT-20k eval | PPO 最优 eval | 涨点 | 当前状态 | 权重保存 |
|------|-------------|--------------|------|---------|---------|
| beat_block_hammer | 57.8% | **86.1%** @step299 | **+28.3pp** ⭐ | 停止于 step335（job超时，2026-06-25）| step100/200/300，57G |
| place_phone_stand | 39.6% | **70.7%** @step299 | **+31.1pp** | 运行中 ~step344 | step50-300，114G |
| place_a2b_left | 35.4% | **51.9%** @step299 | **+16.5pp** | 运行中 ~step327 | step50-300，114G |
| move_can_pot | 52.6% | **63.6%** @step399 | **+11.0pp** | 运行中 ~step459 | step50-450，152G |
| place_a2b_right | 55.1% | **57.1%** @step49 | **+2.0pp** | 运行中 ~step73（早期）| step50，19G |
| place_object_stand | 68.6% | **65.9%** @step299 | **−2.7pp** | 停止于 step319（job超时，2026-06-25）| step100/200/300，57G |
| handover_block | 21.9% | 29.2% @step350 | **−8.3pp** | 放弃（反复崩溃） | step350，19G |
| put_bottles_dustbin | — | — | — | 运行中（Job 10201671，刚启动）| 无 |

> SFT baseline 来自本文档第6节（8 GPU，96 episodes/task）。  
> move_can_pot eval 方差大（32ep/次），最优值 63.6%，最新值 51.9%，建议训练后多轮平均。  
> handover_block 多次失败：v1 训练后 eval 29.2% < SFT，v2 因 collective 通信异常反复崩溃，已放弃。  
> beat_block_hammer 和 place_object_stand 于 2026-06-25 运行，均因 job 超时中止，checkpoints 完好。

### 7.3 主要经验

- **最大涨点**：beat_block_hammer（SFT 57.8%）+28.3pp，place_phone_stand（SFT 39.6%）+31.1pp
- **高 SFT baseline 任务 PPO 提升有限**：place_object_stand（SFT 68.6%）PPO −2.7pp；place_a2b_right（SFT 55.1%）+2pp
- **中等 baseline 任务涨幅大**：place_a2b_left（SFT 35.4%）+16.5pp，move_can_pot（SFT 52.6%）+11pp
- **极稀疏任务 PPO 失败**：handover_block 初始 train success ~9%，标准超参无法收敛
- **eval 单次样本量（32ep）方差大**，需多轮平均才能可靠比较

---

## 8. 已知坏节点（截至 2026-06-24）

在此前实验基础上新增：

| 节点 | 问题 | 首次发现 |
|------|------|---------|
| `SH-IDCA1404-10-140-54-21` | Ray session 冲突 + OOM | moe 5k eval（10131694 等） |
| `SH-IDCA1404-10-140-54-57` | cannot create buffer（OOM） | moe 5k eval click_alarmclock（10130776） |
| `SH-IDCA1404-10-140-54-35` | `Unsupported object type`（进程卡死） | moe 20k eval pick_dual_bottles（10136869） |
| `SH-IDCA1404-10-140-54-88` | SIGKILL / ActorDied | moe 20k eval click_bell（10136865） |

> 完整排除节点列表（含历史）见各 sbatch 文件中的 `--exclude=` 行。

---

## 9. 文件路径索引

| 类型 | 路径前缀 |
|------|---------|
| SFT JAX checkpoints | `.../RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full/` |
| SFT PyTorch checkpoints | `.../RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/` |
| Eval configs (5k) | `examples/embodiment/config/moe_sft5k_eval_{task}.yaml` |
| Eval configs (20k) | `examples/embodiment/config/moe_sft20k_eval_{task}.yaml` |
| PPO configs | `examples/embodiment/config/moe_sft20k_{task}_ppo100.yaml` |
| Eval sbatch (5k) | `examples/embodiment/slurm/moe_sft5k_eval_{task}.sbatch` |
| Eval sbatch (20k) | `examples/embodiment/slurm/moe_sft20k_eval_{task}.sbatch` |
| PPO sbatch | `examples/embodiment/slurm/moe_sft20k_{task}_ppo100.sbatch` |
| Slurm logs | `slurm_logs/moe_sft{5k,20k}_eval_{task}-{jid}.out` |

---

## 10. 5-task BCAM 融合（2026-07-09）

详细记录见：`my_docs/2026-7-9-5task-bcam-merge.md`

融合任务：

- `beat_block_hammer` step50
- `click_alarmclock` step50
- `click_bell` step50
- `move_can_pot` step100
- `move_playingcard_away` step50

搜索范围：每个 task 系数从 `0.3` 到 `0.9`。完整 5D 网格 `7^5=16807` 只做 task-vector Gram matrix 打分，没有全部落盘；实际保存并评估 6 个候选。

最终结果：

| candidate | bbh | ca | cb | mcp | mpa | Avg | Min | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `uniform_a03` | 69.23 | 94.23 | 98.96 | 51.51 | 78.03 | **78.39** | **51.51** | 保留，当前推荐 |
| `uniform_a04` | 70.45 | 100.00 | 100.00 | 42.23 | 76.63 | 77.86 | 42.23 | 保留 |
| `uniform_a05` | 61.34 | 100.00 | 100.00 | 35.70 | 84.16 | 76.24 | 35.70 | 保留 |

已删除低价值候选权重：`best_mean04/best_mean05/best_mean06`。这些候选的结果保存在 `results/5task_bcam_search/eval_success_rates.csv`，可由 `scripts/search_merge_5task_bcam.py` 复现。

### 10.1 后续单 PPO / custom / 6-task 融合对比（2026-07-09）

详细记录见：`my_docs/2026-7-9-ppo-single-vs-merge-followup.md`

关键更新：

| 实验 | 结果 | 判断 |
| --- | --- | --- |
| `custom_0404040505` 5-task | bbh 75.72 / ca 99.00 / cb 100.00 / mcp 38.96 / mpa 85.80，Avg 79.90 | 平均高于原 `uniform_a03/a04/a05`，但 mcp 更弱 |
| 6-task `ca/cb/mpa/bbh/pal/par_a04` | ca 100 / cb 100 / mpa 72.34 / bbh 46.90 / pal 62.12 / par 57.96，Avg 73.22 | 平均高于 SFT，但 bbh 被严重伤害 |
| 单 PPO 矩阵 | `ppo_ca_s50` 与 `ppo_cb_s50` 对 ca/cb 很强，`ppo_mpa_s50` 对 cb/mpa 很强 | ca/cb/mpa/pal/par 更适合融合；bbh 和 mcp 需要谨慎 |

当前判断：`click_alarmclock/click_bell` 是最稳定的可融合任务；`move_playingcard_away/place_a2b_left/place_a2b_right` 可加入但会影响 bbh；`beat_block_hammer` 在 6-task 均匀 `a04` 中明显受损；`move_can_pot` 仍是最不适合硬融合的瓶颈任务。
