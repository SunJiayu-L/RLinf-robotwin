# 2026-07-09 PPO 单权重与后续融合对比

## 目标

在 5-task BCAM 融合后，进一步检查:

1. 5 个单任务 PPO 权重分别在 5 个 task 上的迁移/干扰矩阵。
2. `uniform_a04` 的变体 `custom_0404040505`: `bbh/ca/cb=0.4`, `mcp/mpa=0.5`。
3. 去掉 `move_can_pot` 后，把 `click_alarmclock + click_bell + move_playingcard_away + beat_block_hammer + place_a2b_left + place_a2b_right` 做 6-task 融合。

SFT baseline 使用 `my_docs/2026-7-9-sft20k-eval-20tasks.md` 中的统一 20-task eval。

## 1. 5 PPO x 5 task 单权重矩阵

脚本: `examples/embodiment/slurm/eval_5ppo_on_5task.sbatch`

结果 CSV:

- `results/5ppo_on_5task/latest_success_rates.csv`
- `results/5ppo_on_5task/latest_failures.csv`

当前已完成/补跑进展:

| PPO 权重 | bbh | ca | cb | mcp | mpa | 已完成均值 | 已完成均值涨点 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ppo_bbh_s50` | 71.09 | 72.46 | 41.67 | pending/timeout | 69.68 | 63.73 | -1.27 |
| `ppo_ca_s50` | 72.79 | 100.00 | 100.00 | timeout | 77.94 | 87.68 | +17.47 |
| `ppo_cb_s50` | 71.01 | 100.00 | 98.96 | 25.84 | 72.68 | 73.70 | +16.29 |
| `ppo_mcp_s100` | 55.14 | 71.43 | 53.13 | 67.55 | 68.86 | 63.22 | -2.27 |
| `ppo_mpa_s50` | 58.49 | 86.71 | 98.96 | 48.89 | 90.07 | 76.62 | +14.08 |

说明:

- `ppo_bbh_s50/move_can_pot` 补跑 job `10212550` 仍在运行；主 job 已 timeout。
- `ppo_ca_s50/move_can_pot` 主 job timeout，暂未补跑成功。
- `ppo_cb_s50/move_can_pot` 补跑 `10212549` 成功，为 `25.84%`。
- `ppo_mcp_s100/click_bell` 补跑 `10212551` 成功，为 `53.13%`。

观察:

1. `ppo_ca_s50` 和 `ppo_cb_s50` 对 `click_alarmclock/click_bell` 都很强，且对 `beat_block_hammer` 基本不伤。
2. `ppo_mpa_s50` 对 `click_bell` 和自身 `move_playingcard_away` 很强，但伤 `beat_block_hammer`。
3. `move_can_pot` 是跨任务迁移最差的方向: `ppo_cb_s50` 到 mcp 只有 `25.84%`，`ppo_mpa_s50` 到 mcp `48.89%`，`ppo_ca_s50/bbh_s50` 的 mcp eval 都容易 timeout。

## 2. 5-task custom_0404040505

用户提出从 `uniform_a04` 出发，把后三个中的 `mcp/mpa` 提到 `0.5`:

| bbh | ca | cb | mcp | mpa |
| ---: | ---: | ---: | ---: | ---: |
| 0.4 | 0.4 | 0.4 | 0.5 | 0.5 |

权重:

`/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_5task_bcam_custom_0404040505/full_weights.pt`

Eval job: `10212420`

| candidate | bbh | ca | cb | mcp | mpa | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `custom_0404040505` | 75.72 | 99.00 | 100.00 | 38.96 | 85.80 | 79.90 |
| `uniform_a03` | 69.23 | 94.23 | 98.96 | 51.51 | 78.03 | 78.39 |
| `uniform_a04` | 70.45 | 100.00 | 100.00 | 42.23 | 76.63 | 77.86 |
| `uniform_a05` | 61.34 | 100.00 | 100.00 | 35.70 | 84.16 | 76.24 |

结论:

- `custom_0404040505` 当前 5-task 平均最高 `79.90%`。
- 它显著改善 `bbh` 和 `mpa`，但 `mcp` 进一步低于 `uniform_a04`，只有 `38.96%`。
- 如果评价看平均值，它优于之前 5-task 候选；如果评价看最差任务，`uniform_a03` 仍更稳。

## 3. 6-task 融合: ca/cb/mpa/bbh/pal/par

用户提出组合:

- `click_alarmclock`
- `click_bell`
- `move_playingcard_away`
- `beat_block_hammer`
- `place_a2b_left`
- `place_a2b_right`

本次先做均匀系数 `a=0.4`:

| task | step | alpha |
| --- | ---: | ---: |
| click_alarmclock | 50 | 0.4 |
| click_bell | 50 | 0.4 |
| move_playingcard_away | 50 | 0.4 |
| beat_block_hammer | 50 | 0.4 |
| place_a2b_left | 50 | 0.4 |
| place_a2b_right | 50 | 0.4 |

权重:

`/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_6task_ca_cb_mpa_bbh_pal_par_a04/full_weights.pt`

脚本:

- merge: `examples/embodiment/slurm/merge_6task_ca_cb_mpa_bbh_pal_par_a04.sbatch`
- eval: `examples/embodiment/slurm/eval_custom_ckpt_6task_ca_cb_mpa_bbh_pal_par.sbatch`

Eval job: `10212537`

结果 CSV:

- `results/6task_ca_cb_mpa_bbh_pal_par_merge/eval_success_rates.csv`
- `results/6task_ca_cb_mpa_bbh_pal_par_merge/eval_vs_sft_baseline.csv`

| task | merge success | SFT-20k baseline | delta |
| --- | ---: | ---: | ---: |
| click_alarmclock | 100.00 | 80.60 | +19.40 |
| click_bell | 100.00 | 39.60 | +60.40 |
| move_playingcard_away | 72.34 | 67.60 | +4.74 |
| beat_block_hammer | 46.90 | 72.20 | -25.30 |
| place_a2b_left | 62.12 | 48.20 | +13.92 |
| place_a2b_right | 57.96 | 56.60 | +1.36 |
| AVG | 73.22 | 60.80 | +12.42 |

结论:

1. 这个 6-task merge 平均比 SFT 高 `+12.42pp`。
2. `click_alarmclock/click_bell` 非常稳定，都到 `100%`。
3. `place_a2b_left` 有明显提升，`place_a2b_right` 只有小幅提升。
4. `beat_block_hammer` 被严重伤害，降到 `46.90%`。这说明 `bbh` 不适合直接以 `a=0.4` 混入这个 6-task 组，或者需要显著提高 bbh 自身系数/降低其他冲突项。

## 4. 当前判断

适合融合:

- `click_alarmclock`
- `click_bell`
- `move_playingcard_away`
- `place_a2b_left`
- `place_a2b_right`

需要谨慎:

- `beat_block_hammer`: 单独 PPO 和 5-task custom 能保持，但 6-task `a04` 中被严重伤害。
- `move_can_pot`: 目前是最不适合硬融合的 task，容易被其他 task vector 破坏。

下一步建议:

1. 对 6-task 组合做非均匀系数，而不是均匀 `a04`。例如提高 bbh 到 `0.6/0.7`，同时保持 ca/cb/pal/par/mpa 在 `0.3/0.4`。
2. 如果目标是平均成功率，`custom_0404040505` 已经比 `uniform_a03/a04/a05` 更好。
3. 如果目标是最差任务不崩，仍应优先围绕 `uniform_a03` 或专门处理 `move_can_pot`。
