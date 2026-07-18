# 2026-07-09 5-task BCAM 权重融合实验记录

## 目标

对 5 个 RoboTwin task 的 PI0.5 MoE PPO checkpoint 做 task arithmetic 融合，并从系数 `0.3..0.9` 搜索可用候选。

任务与 checkpoint:

| short | task | step | checkpoint |
| --- | --- | ---: | --- |
| bbh | beat_block_hammer | 50 | `/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260707-22:22:21-moe_sft20k_beat_block_hammer_ppo100/moe_sft20k_beat_block_hammer_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt` |
| ca | click_alarmclock | 50 | `/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260708-13:50:33-moe_sft20k_click_alarmclock_ppo100/moe_sft20k_click_alarmclock_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt` |
| cb | click_bell | 50 | `/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260708-13:56:28-moe_sft20k_click_bell_ppo100/moe_sft20k_click_bell_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt` |
| mcp | move_can_pot | 100 | `/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260706-23:28:23-moe_sft20k_move_can_pot_ppo100/moe_sft20k_move_can_pot_ppo100/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt` |
| mpa | move_playingcard_away | 50 | `/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100/moe_sft20k_move_playingcard_away_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt` |

Base model:

`/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors`

## 搜索方法

脚本: `scripts/search_merge_5task_bcam.py`

完整网格是 `7^5 = 16807` 个候选，系数集合为 `{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}`。由于磁盘空间接近 quota，没有落盘全部网格，而是:

1. 只在 `PI05_PPO_TRAINABLE_PREFIXES` 对应参数上构造 task vector。
2. 计算 5 个 task vector 的 Gram matrix。
3. 用 `alpha^T G alpha` 对完整网格做近似打分。
4. 只落盘 6 个候选: 3 个均匀系数，3 个固定 mean alpha 下 `norm_per_alpha_sum` 最小的非均匀候选。

搜索输出:

- `results/5task_bcam_search/task_vector_gram.csv`
- `results/5task_bcam_search/top200_by_norm_per_alpha.csv`
- `results/5task_bcam_search/selected_candidates.csv`

Gram matrix 主要观察:

| task | self norm |
| --- | ---: |
| bbh | 3.5209 |
| ca | 2.4670 |
| cb | 5.5232 |
| mcp | 4.9628 |
| mpa | 2.4119 |

`click_bell` 和 `move_can_pot` 的 task vector norm 明显更大；非均匀搜索因此倾向降低 `cb/mcp` 系数。但后续实际 eval 显示，这个 norm-based 策略没有带来更好的 5-task 平衡。

## 候选权重

原始落盘 6 个候选，每份约 6.8G:

| candidate | bbh | ca | cb | mcp | mpa | 处理 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| best_mean04 | 0.4 | 0.5 | 0.3 | 0.3 | 0.5 | 已删除 |
| best_mean05 | 0.5 | 0.6 | 0.3 | 0.4 | 0.7 | 已删除 |
| best_mean06 | 0.6 | 0.8 | 0.4 | 0.4 | 0.8 | 已删除 |
| uniform_a03 | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 保留 |
| uniform_a04 | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 | 保留 |
| uniform_a05 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 保留 |

保留权重:

- `/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_5task_bcam_uniform_a03/full_weights.pt`
- `/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_5task_bcam_uniform_a04/full_weights.pt`
- `/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_5task_bcam_uniform_a05/full_weights.pt`

删除原因: 3 个 `best_mean` 候选的平均成功率和最低任务成功率都低于均匀候选，尤其 `move_can_pot` 明显退化。结果 CSV 已保存，因此可按脚本复现。

## Eval 配置

脚本: `examples/embodiment/slurm/eval_5task_bcam_candidates.sbatch`

主评估 job:

- `10211960`: `best_mean04`, `best_mean05`
- `10211961`: `best_mean06`, `uniform_a03`
- `10211962`: `uniform_a04`, `uniform_a05`

补跑 job:

| job | candidate/task | 原因 | 结果 |
| --- | --- | --- | --- |
| 10212228 | best_mean05/move_playingcard_away | 主 job 1800s timeout | success 0.8888889 |
| 10212229 | uniform_a04/beat_block_hammer | 主 job 1800s timeout | success 0.70445347 |
| 10212230 | best_mean06/move_can_pot | 主 job 1800s timeout | success 0.1904762 |

最终 CSV:

- `results/5task_bcam_search/eval_success_rates.csv`
- `results/5task_bcam_search/eval_failures_resolved.csv`

## 结果

| candidate | bbh | ca | cb | mcp | mpa | Avg | Min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_a03 | 69.23 | 94.23 | 98.96 | 51.51 | 78.03 | 78.39 | 51.51 |
| uniform_a04 | 70.45 | 100.00 | 100.00 | 42.23 | 76.63 | 77.86 | 42.23 |
| uniform_a05 | 61.34 | 100.00 | 100.00 | 35.70 | 84.16 | 76.24 | 35.70 |
| best_mean04 | 71.18 | 98.08 | 100.00 | 29.37 | 80.71 | 75.87 | 29.37 |
| best_mean05 | 63.11 | 95.00 | 100.00 | 25.48 | 88.89 | 74.50 | 25.48 |
| best_mean06 | 50.13 | 95.00 | 100.00 | 19.05 | 68.31 | 66.50 | 19.05 |

结论:

1. 当前最推荐 `uniform_a03`: 平均成功率最高 `78.39%`，并且最低任务 `move_can_pot=51.51%` 是所有候选里最高的。
2. `uniform_a04` 也可作为备选: bbh/ca/cb 很强，平均只比 `uniform_a03` 低 `0.53` 个百分点，但 mcp 降到 `42.23%`。
3. `uniform_a05` 对 mpa 更好，但 mcp 继续下降，说明 5-task 总体系数过高会损伤 move_can_pot。
4. norm-based 非均匀候选没有胜过均匀系数；后续如果继续搜索，应优先在 `uniform_a03` 附近局部微调，而不是提高总系数。

## 下一步建议

局部搜索建议从 `uniform_a03` 出发:

- 固定 `bbh/ca/cb/mpa = 0.3`，搜索 `mcp = 0.2, 0.25, 0.3, 0.35, 0.4`。
- 或搜索小范围 5D 网格 `{0.25, 0.3, 0.35}`，只落盘 top 候选。
- 评估指标建议优先看 `min_success`，其次看 `avg_success`；当前瓶颈是 `move_can_pot`，单纯提高均值可能掩盖 mcp 退化。

## 空间记录

用户 quota 初始约 `15.79 TiB / 16.15 TiB`。本次先清理历史 3-task 临时 grid 权重，再只保存 6 个 5-task 候选。评估后删除 3 个低价值非均匀候选，最终只保留 3 个均匀候选。

清理后 quota:

`15.75 TiB / 16.15 TiB`
