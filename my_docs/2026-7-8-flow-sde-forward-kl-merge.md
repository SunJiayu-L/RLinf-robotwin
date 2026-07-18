# Flow-SDE 正向 KL 融合诊断实验记录

**日期**：2026-07-08  
**目标**：把 RL's Razor 的正向 KL 思路接入 pi-RL / Flow-SDE 权重融合流程，用作融合前的抗遗忘风险诊断，并据此指导 task arithmetic 的 alpha 选择。  
**相关代码**：`scripts/compute_forward_kl_merge.py`，`examples/embodiment/slurm/compute_forward_kl_merge.sbatch`  
**第一版输出目录**：`results/forward_kl_merge/`（单条 base trajectory，保留作对照）  
**最终输出目录**：`results/forward_kl_merge_sde20/`（20 条 base Flow-SDE trajectory，后续以此为准）

---

## 1. 背景结论

从 pi-RL paper 和代码实现看，当前 OpenPI 动作模型在 `noise_method: flow_sde` 下把 denoising transition 写成高斯分布：

$$
p(a_{t-\Delta t}\mid a_t,\mathrm{obs})
=
\mathcal{N}\!\left(
a_{t-\Delta t};
\mu_\theta(a_t,\mathrm{obs},t),
\sigma^2(t)I
\right)
$$

代码位置：

- `rlinf/models/embodiment/openpi/openpi_action_model.py`
- `sample_mean_var_val()` 给出 Flow-SDE 的 transition mean/std
- `get_logprob_norm()` 对高斯 transition 计算 logprob
- `sample_actions()` 在 PPO rollout 中保存 denoising chain 和 logprob

注意这里的“一致”指的是单步 transition 参数：手写 mean/std 和
`sample_mean_var_val(..., sample_method="flow_sde")` 的 Flow-SDE 分支公式一致。
它不是说离线 KL 脚本完全复现了 PPO rollout 的完整随机 chain 分布。默认
`joint_logprob=False` 时，PPO rollout 只随机选择一个 denoise step 注入 SDE，
其余 step 走 ODE；而完整 chain 还包含连续初始噪声和 SDE 噪声，不能被穷尽，
只能用 Monte Carlo 采样估计。

RL's Razor 的核心观察是：旧任务遗忘可以用新任务分布上的正向 KL 预测：

$$
\mathbb{E}_{x \sim \mathcal{D}_{\mathrm{new}}}
\left[
D_{\mathrm{KL}}
\left(
\pi_0(\cdot \mid x)
\,\|\, 
\pi_{\mathrm{candidate}}(\cdot \mid x)
\right)
\right]
$$

这个方向是 $pi_0 || pi_candidate$，不是 PPO 训练里常见的 sampled reverse/old-policy penalty。Flow-SDE 的单步 transition 是高斯，因此在给定 $(o, x_t, t)$ 时可以直接做 analytic KL。完整连续 chain 分布不能穷尽，本实验用 base trajectory 上的单步 analytic KL 做离线风险诊断。

本次采用的融合诊断指标：

$$
D_{\mathrm{KL}}^{\mathrm{forward}}
=
\mathbb{E}_{(o,t,d)}
\left[
D_{\mathrm{KL}}
\left(
\mathcal{N}\!\left(\mu_{0}^{(d)}(o,t), \sigma_{0}^{(d)}(t)^2\right)
\,\|\, 
\mathcal{N}\!\left(\mu_{\mathrm{cand}}^{(d)}(o,t), \sigma_{\mathrm{cand}}^{(d)}(t)^2\right)
\right)
\right]
$$


其中 `base` 是 SFT-20k / pi0.5 MoE demo clean，`candidate` 是单任务 PPO 或融合权重。最终版按 20 个 seed 从 base Flow-SDE 中采样 trajectory，并在相同的 base `x_t` 上比较 base 和 candidate 的 Flow-SDE transition。这里采样的是 base chain 上的 probe 点，不是对 candidate 自身闭环 rollout chain 做解析积分。

---

## 2. 实现与运行

### 实现内容

新增脚本 `scripts/compute_forward_kl_merge.py`：

- 读取真实 RoboTwin observation，复用 `plot_euler_trajectory_heatmaps.py::load_real_obs_by_name`
- 对每个任务构造 base trajectory
- 对 base / candidate 分别计算 Flow-SDE Gaussian transition
- 用高斯解析式计算 `KL(base || candidate)`
- 输出 per-task rows、candidate summary 和 markdown summary
- 支持两类 candidate：
  - 已经落盘的旧 pairwise / 5task / 新 step50 融合权重
  - 内存中直接 task arithmetic 生成的 `bbh/pal/par` step50 alpha grid

新增 sbatch：

- `examples/embodiment/slurm/compute_forward_kl_merge.sbatch`
- 1 GPU，batch size 4，默认输出到 `results/forward_kl_merge/`

### 运行过程

| Job ID | 结果 | 说明 |
|---|---|---|
| 10208887 | failed | `sample_mean_var_val()` 内部 timestep dtype 为 fp32，和 bf16 权重不匹配 |
| 10208920 | failed | `get_velocity()` 后 suffix projection 输入 dtype 与 bf16 权重不匹配 |
| 10208958 | completed | 修正 dtype 后完成，耗时约 795s |
| 10209328 | completed | 改为 20 条 base Flow-SDE trajectory 后重算，20:17:34 - 20:36:10 |

修正方式：

- 不直接调用 `sample_mean_var_val()`，改为显式构造 bf16 timestep 并手动计算 Flow-SDE mean/std
- 在 `action_out_proj` 前将 suffix hidden cast 到 projection weight dtype

手写 mean/std 对照 `sample_mean_var_val()` 的 Flow-SDE 分支：

- `sigma_i = noise_level * sqrt(t / (1 - t))`，首个 `t=1` 时用下一个 timestep 避免除零
- `x0_pred = x_t - v_t * t`
- `x1_pred = x_t + v_t * (1 - t)`
- `x0_weight = 1 - (t - delta)`
- `x1_weight = (t - delta) - sigma_i^2 * delta / (2t)`
- `std = sqrt(delta) * sigma_i`

这和 rollout 单步 Flow-SDE transition 完全一致；差异只在评估分布：本脚本在 base trajectory 的每个 probe step 上算 KL，而默认 PPO rollout 每条样本只随机选择一个 SDE denoise step。

最终输出：

- `results/forward_kl_merge/forward_kl_rows.csv`
- `results/forward_kl_merge/forward_kl_candidate_summary.csv`
- `results/forward_kl_merge/summary.md`
- `slurm_logs/compute_forward_kl_merge-10208958.out`
- `results/forward_kl_merge_sde20/forward_kl_rows.csv`
- `results/forward_kl_merge_sde20/forward_kl_candidate_summary.csv`
- `results/forward_kl_merge_sde20/summary.md`
- `slurm_logs/compute_forward_kl_merge-10209328.out`

限制：

- 本次 batch size 为 4，最终版每个任务采 20 条 base Flow-SDE trajectory，是比单轨迹更稳定的离线筛选，但仍不是最终统计显著性实验
- 20 条 trajectory 是 Monte Carlo 近似，不能穷尽完整连续 chain；能穷尽的只有离散 denoise index
- `pps` 的旧 best solo checkpoint 没找到同一套旧基线路径，只保留绝对 KL，`ratio_to_solo` 为 `nan`
- 正向 KL 是抗遗忘风险指标，不直接等价于 eval success，也不能替代真实环境评测

---

## 3. 单任务 KL 基线

`KL/dim` 是针对某一个 task 的 observation 分布计算的 per-task 指标，不是
candidate 的全局唯一数值。对一个三任务融合候选，必须分别在三个任务的
observation 上计算：

```
candidate @ beat_block_hammer obs
candidate @ place_a2b_left obs
candidate @ place_a2b_right obs
```

summary 表里的 `Avg KL/dim` 和 `Max KL/dim` 只是这些 per-task rows 的聚合。
判断遗忘风险时应优先看 `Max KL/dim` / `Max ratio_to_solo`，再看平均值；
只看平均值可能掩盖某个 task 的单点崩坏。

`ratio_to_solo` 的定义：

```
ratio_to_solo(task) =
    KL(base || merged_candidate on task obs)
    / KL(base || solo_PPO_task on task obs)
```

越接近或低于 1，说明融合权重相对单任务 PPO 没有额外偏离 base 太多。对于 step50 单任务，solo KL 本身很小，因此 ratio 容易被放大，需要同时看绝对 `KL/dim`。

| Solo key | Task | KL/dim |
|---|---|---:|
| old_bbh | beat_block_hammer | 0.011240 |
| old_pal | place_a2b_left | 0.007568 |
| old_par | place_a2b_right | 0.003912 |
| old_mcp | move_can_pot | 0.010529 |
| bbh50 | beat_block_hammer | 0.001262 |
| pal50 | place_a2b_left | 0.001856 |
| pal100 | place_a2b_left | 0.001750 |
| par50 | place_a2b_right | 0.003912 |
| par200 | place_a2b_right | 0.010555 |
| mcp50 | move_can_pot | 0.001099 |
| mcp100 | move_can_pot | 0.002526 |
| mpa50 | move_playingcard_away | 0.001485 |
| mpa100 | move_playingcard_away | 0.003119 |
| mpa150 | move_playingcard_away | 0.004882 |

---

## 4. 已有融合权重结果

### Candidate summary

按 `max_ratio_to_solo` 从低到高排序，越靠前越保守。
这里每个 candidate 会先对它包含的所有 task 分别计算 `KL/dim`，再汇总出
`Avg KL/dim` 和 `Max KL/dim`。例如 `bbh+pal+par` 的三任务融合不是只算一个
KL，而是分别算 bbh、pal、par 三行；`Max` 代表最坏任务的风险。

| Candidate | Tasks | Avg KL/dim | Max KL/dim | Avg ratio | Max ratio |
|---|---|---:|---:|---:|---:|
| pair_pal_mcp_a07 | pal+mcp | 0.004398 | 0.005176 | 0.514 | 0.684 |
| pair_pps_pal_a07 | pps+pal | 0.005380 | 0.005457 | 0.701 | 0.701 |
| pair_bbh_pps_a07 | bbh+pps | 0.016606 | 0.023855 | 0.832 | 0.832 |
| pair_pps_mcp_a07 | pps+mcp | 0.013255 | 0.014945 | 1.098 | 1.098 |
| pair_pal_par_a07 | pal+par | 0.004102 | 0.004682 | 0.831 | 1.197 |
| pair_bbh_pal_a07 | bbh+pal | 0.008386 | 0.011323 | 0.990 | 1.496 |
| pair_pps_par_a07 | pps+par | 0.008118 | 0.010357 | 1.503 | 1.503 |
| pair_par_mcp_a07 | par+mcp | 0.007499 | 0.008088 | 1.267 | 1.766 |
| pair_bbh_mcp_a07 | bbh+mcp | 0.018454 | 0.022036 | 1.708 | 2.093 |
| s50_pal07_par07 | pal+par | 0.004323 | 0.005557 | 1.892 | 2.994 |
| s50_pal07_par07_mpa07 | pal+par+mpa | 0.004753 | 0.006514 | 2.410 | 3.510 |
| s50_bbh07_pal07_par07 | bbh+pal+par | 0.004445 | 0.006888 | 2.376 | 3.712 |
| s50_bbh09_pal07_par07 | bbh+pal+par | 0.004911 | 0.007806 | 2.641 | 4.206 |
| s50_mpa_pal_par_mcp_a07 | mpa+pal+par+mcp | 0.005934 | 0.007952 | 2.780 | 4.285 |
| s50_pal08_par08_mpa07 | pal+par+mpa | 0.005905 | 0.008112 | 2.993 | 4.371 |
| old_5task_a07 | bbh+pps+pal+par+mcp | 0.036610 | 0.048739 | 4.608 | 6.414 |
| old_5task_chain_bbh_a07 | bbh+pps+pal+par+mcp | 0.056823 | 0.077582 | 7.290 | 10.301 |

### 和已有 eval 的对应关系

1. 旧 5task collapse 被正向 KL 明确捕捉到。
   - `old_5task_a07` max ratio 6.414
   - `old_5task_chain_bbh_a07` max ratio 10.301
   - 这和之前 5task eval 中几乎所有任务显著低于单任务 PPO 一致。

2. 链式融合比均匀 5task 更差，也被 KL 捕捉到。
   - 均匀 5task avg KL/dim 0.036610
   - chain 5task avg KL/dim 0.056823
   - chain 的 pal/par ratio 都超过 10，说明把 bbh 作为锚点没有保护 bbh，反而整体偏离更大。

3. `bbh+mcp` 是 pairwise 中高风险组合。
   - max ratio 2.093
   - 旧 eval 中 `bbh+mcp` 的 bbh 从 86.1% 掉到 39.9%，是全场最差 pair。

4. `pal+par` 是较稳的 pair。
   - 旧 best pair `pair_pal_par_a07` max ratio 1.197
   - 旧 eval 中 pal 明显提升，par 基本不掉。

5. KL 不是充分条件。
   - `pair_pal_mcp_a07` KL 很低，但旧 eval 中 mcp 仍明显下降。
   - 说明正向 KL 适合作为遗忘风险过滤器，但不能单独判断任务 reward 是否保留或是否有正迁移。

---

## 5. Step50 bbh/pal/par alpha grid

本次额外在内存中扫描了 step50 三任务：

```
bbh alpha in {0.5, 0.7, 0.9}
pal alpha in {0.5, 0.7}
par alpha in {0.5, 0.7}
```

结果按 max ratio 从低到高：

每个 grid candidate 都在 bbh、pal、par 三个任务 observation 上分别计算 KL。
表中的 `Avg` 是三行平均，`Max` 是三行最大值；因此 `Max ratio` 更接近
“这组三任务融合里最可能遗忘的任务”的风险信号。

| Candidate | bbh | pal | par | Avg KL/dim | Max KL/dim | Avg ratio | Max ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| grid_s50_bbh05_pal05_par05 | 0.5 | 0.5 | 0.5 | 0.002268 | 0.003498 | 1.222 | 1.885 |
| grid_s50_bbh07_pal05_par05 | 0.7 | 0.5 | 0.5 | 0.002611 | 0.004064 | 1.409 | 2.190 |
| grid_s50_bbh05_pal07_par05 | 0.5 | 0.7 | 0.5 | 0.002899 | 0.004378 | 1.567 | 2.359 |
| grid_s50_bbh09_pal05_par05 | 0.9 | 0.5 | 0.5 | 0.003072 | 0.004835 | 1.651 | 2.605 |
| grid_s50_bbh07_pal07_par05 | 0.7 | 0.7 | 0.5 | 0.003212 | 0.004871 | 1.723 | 2.625 |
| grid_s50_bbh05_pal05_par07 | 0.5 | 0.5 | 0.7 | 0.003491 | 0.005636 | 1.870 | 3.037 |
| grid_s50_bbh07_pal05_par07 | 0.7 | 0.5 | 0.7 | 0.003677 | 0.005821 | 1.974 | 3.137 |
| grid_s50_bbh09_pal07_par05 | 0.9 | 0.7 | 0.5 | 0.003862 | 0.005828 | 2.081 | 3.140 |
| grid_s50_bbh05_pal07_par07 | 0.5 | 0.7 | 0.7 | 0.004085 | 0.006243 | 2.184 | 3.364 |
| grid_s50_bbh09_pal05_par07 | 0.9 | 0.5 | 0.7 | 0.004133 | 0.006656 | 2.222 | 3.586 |
| grid_s50_bbh07_pal07_par07 | 0.7 | 0.7 | 0.7 | 0.004454 | 0.006960 | 2.390 | 3.750 |
| grid_s50_bbh09_pal07_par07 | 0.9 | 0.7 | 0.7 | 0.004911 | 0.007806 | 2.641 | 4.206 |

关键观察：

- 当前已评测的 `0.7/0.7/0.7` 不是 KL 最稳点，max ratio 约 3.75。
- `0.9/0.7/0.7` 会继续增加 bbh 与 pal 的 KL 负担，max ratio 约 4.21。
- 这和 eval 现象一致：提高 bbh alpha 后，bbh 从 60.9% 回升到 73.7%，但 pal 从 71.0% 掉到 55.5%。
- `par` 的自身 ratio 一直不高，但把 `par alpha` 从 0.5 提到 0.7 会明显推高 bbh/pal 上的 KL，因此如果 par 已经够强，优先保持 par alpha 低一点。
- 最保守的三任务候选是 `0.5/0.5/0.5`，max KL/dim 0.003498，max ratio 1.885。

---

## 6. 对 alpha 选择的指导

本次实验给出的 practical rule：

### 6.1 两个指标一起看

1. `max_ratio_to_solo`
   - 衡量融合相对单任务 PPO 是否额外偏离 base
   - 适合不同任务之间做相对比较

2. `max_KL/dim`
   - 衡量绝对偏离
   - 对 step50 尤其重要，因为 step50 solo KL 很小，ratio 会被放大

### 6.2 当前建议阈值

这些阈值只针对本次 pi0.5 Flow-SDE / RoboTwin / batch=4 诊断，不应直接外推为通用定律。

| 风险区间 | max ratio | max KL/dim | 用途 |
|---|---:|---:|---|
| 保守 | <= 1.5 | <= 0.0035 | 优先进入 eval |
| 中等 | 1.5 - 2.5 | 0.0035 - 0.006 | 可作为探索候选 |
| 高风险 | > 3.0 | > 0.006 | 除非有明确单任务优先级，否则不建议优先 eval |

### 6.3 多任务融合 alpha 原则

- N >= 3 时，不要默认从 alpha 0.7 起步；更合理的是先扫 0.5，再加少量任务特异调整。
- 对 bbh/pal/par step50，下一批更值得 eval 的候选：
  - `bbh=0.5, pal=0.5, par=0.5`：最保守，适合验证 KL filter 是否提升平均稳定性
  - `bbh=0.7, pal=0.5, par=0.5`：给 bbh 一些额外权重，同时控制 pal/par 的 KL
  - `bbh=0.5, pal=0.7, par=0.5`：如果更看重 pal，可作为对照
- 暂不建议继续优先扩大 `bbh=0.9, pal=0.7, par=0.7` 这一类候选；它提升 bbh 的代价已经在 pal 上体现。
- 对 `pal/par/mpa` 和 `mpa/pal/par/mcp`，现有 alpha 0.7/0.8 的 max ratio 已经在 3.5-4.4 区间，下一步应先降 alpha，而不是直接加更多任务。

---

## 7. 如何把正向 KL 加到权重融合流程

当前建议不是把 KL 作为训练 loss，而是作为 task arithmetic 后的离线筛选器：

1. 准备每个任务的 solo PPO checkpoint 和 observation cache
2. 生成 alpha grid 或随机候选
3. 对每个候选计算每个任务上的 `KL(base || candidate)`
4. 按以下规则过滤：
   - 先过滤 `max_ratio_to_solo > 3`
   - 再过滤 `max_KL/dim > 0.006`
   - 从剩余候选中优先选 avg KL 低、任务优先级满足要求的 top-K
5. 只对 top-K 做真实环境 eval

伪流程：

```
for alpha_vector in alpha_grid:
    candidate = base + sum_i alpha_i * (ppo_i - base)
    fkl_rows = compute_flow_sde_forward_kl(base, candidate, task_obs)
    if max_ratio <= threshold_ratio and max_kl <= threshold_abs:
        keep_for_eval(candidate)
```

如果后续要自动化，可以把 `scripts/compute_forward_kl_merge.py` 和 `scripts/task_arithmetic_rlinf.py` 进一步合并成：

```
task_arithmetic_rlinf.py --scan-alphas ... --kl-filter ...
```

但目前保持独立脚本更方便调试和复用已有 observation / HiddenSim 分析代码。

---

## 8. 本次结论

正向 KL 对“融合是否会遗忘”有明显解释力：

- 能解释旧 5task 和 chain 方案为什么 collapse
- 能识别 `bbh+mcp` 这类高冲突 pair
- 能解释 `bbh alpha` 提高后 bbh 回升但 pal 下降
- 能为下一轮 alpha grid 提供比盲目 eval 更便宜的筛选依据

但它不是完整的融合质量指标：

- 低 KL 不保证 reward 提升
- 低 KL 不保证任务间正迁移
- batch=4 的结果适合作为快速 filter，最终仍需真实 eval 确认

下一步建议：

1. 先 eval `bbh/pal/par` 的低 KL 三个候选：`0.5/0.5/0.5`、`0.7/0.5/0.5`、`0.5/0.7/0.5`
2. 对 `pal/par/mpa` 和 `mpa/pal/par/mcp` 降 alpha 后重算 forward-KL，再决定是否 eval
3. 补齐 `pps` old solo checkpoint 路径后重跑一次旧 pairwise summary，让 pps 的 ratio 可比较
4. 将 forward-KL、HiddenSim、真实 eval 三者放在同一张表里，检查哪些指标能预测不同类型的失败

---

## 9. SDE-20 重算结果（最终采用）

根据新的计算定义：

```
for obs in task_obs:
    for seed in seeds:  # 20 seeds, 42-61
        sample x_trajectory from base Flow-SDE
        for each t:
            compute KL(base transition || candidate transition)
```

脚本参数：

```
python scripts/compute_forward_kl_merge.py \
    --output-dir results/forward_kl_merge_sde20 \
    --device cuda \
    --batch-size 4 \
    --num-trajectories 20 \
    --seed-start 42
```

### 9.1 SDE-20 单任务 KL 基线

| Solo key | Task | KL/dim |
|---|---|---:|
| old_bbh | beat_block_hammer | 0.011160 |
| old_pal | place_a2b_left | 0.008249 |
| old_par | place_a2b_right | 0.004069 |
| old_mcp | move_can_pot | 0.010067 |
| bbh50 | beat_block_hammer | 0.001290 |
| pal50 | place_a2b_left | 0.001701 |
| pal100 | place_a2b_left | 0.001632 |
| par50 | place_a2b_right | 0.004069 |
| par200 | place_a2b_right | 0.010810 |
| mcp50 | move_can_pot | 0.001262 |

### 9.2 SDE-20 candidate summary

| Candidate | Tasks | Avg KL/dim | Max KL/dim | Avg ratio | Max ratio |
|---|---|---:|---:|---:|---:|
| pair_pal_mcp_a07 | pal+mcp | 0.004255 | 0.004844 | 0.476 | 0.587 |
| pair_pps_pal_a07 | pps+pal | 0.005565 | 0.005662 | 0.663 | 0.663 |
| pair_bbh_pps_a07 | bbh+pps | 0.015772 | 0.022215 | 0.836 | 0.836 |
| pair_pps_mcp_a07 | pps+mcp | 0.012733 | 0.014455 | 1.094 | 1.094 |
| pair_pal_par_a07 | pal+par | 0.004480 | 0.005153 | 0.864 | 1.266 |
| pair_bbh_pal_a07 | bbh+pal | 0.008516 | 0.011264 | 0.941 | 1.365 |
| pair_pps_par_a07 | pps+par | 0.008244 | 0.010465 | 1.480 | 1.480 |
| pair_par_mcp_a07 | par+mcp | 0.006861 | 0.007388 | 1.145 | 1.557 |
| grid_s50_bbh05_pal05_par05 | bbh+pal+par | 0.002293 | 0.003274 | 1.262 | 1.924 |
| pair_bbh_mcp_a07 | bbh+mcp | 0.017166 | 0.019914 | 1.635 | 1.978 |
| grid_s50_bbh07_pal05_par05 | bbh+pal+par | 0.002604 | 0.003608 | 1.439 | 2.120 |
| pair_bbh_par_a07 | bbh+par | 0.007627 | 0.008995 | 1.386 | 2.211 |
| grid_s50_bbh05_pal07_par05 | bbh+pal+par | 0.002890 | 0.003971 | 1.592 | 2.334 |
| grid_s50_bbh09_pal05_par05 | bbh+pal+par | 0.003044 | 0.004169 | 1.671 | 2.450 |
| grid_s50_bbh07_pal07_par05 | bbh+pal+par | 0.003217 | 0.004332 | 1.773 | 2.546 |
| grid_s50_bbh09_pal07_par05 | bbh+pal+par | 0.003811 | 0.005010 | 2.090 | 2.945 |
| s50_pal07_par07 | pal+par | 0.004134 | 0.005024 | 1.875 | 2.953 |
| grid_s50_bbh05_pal05_par07 | bbh+pal+par | 0.003458 | 0.005180 | 1.898 | 3.044 |
| grid_s50_bbh07_pal05_par07 | bbh+pal+par | 0.003707 | 0.005330 | 2.042 | 3.133 |
| grid_s50_bbh05_pal07_par07 | bbh+pal+par | 0.004060 | 0.005720 | 2.227 | 3.362 |
| grid_s50_bbh09_pal05_par07 | bbh+pal+par | 0.004133 | 0.005960 | 2.282 | 3.503 |
| s50_pal07_par07_mpa07 | pal+par+mpa | 0.004709 | 0.006204 | 2.473 | 3.646 |
| grid_s50_bbh07_pal07_par07 | bbh+pal+par | 0.004433 | 0.006289 | 2.442 | 3.696 |
| s50_bbh07_pal07_par07 | bbh+pal+par | 0.004450 | 0.006321 | 2.450 | 3.715 |
| grid_s50_bbh09_pal07_par07 | bbh+pal+par | 0.004938 | 0.007009 | 2.721 | 4.119 |
| s50_bbh09_pal07_par07 | bbh+pal+par | 0.004938 | 0.007009 | 2.721 | 4.119 |
| s50_pal08_par08_mpa07 | pal+par+mpa | 0.005888 | 0.007882 | 3.097 | 4.633 |
| s50_mpa_pal_par_mcp_a07 | mpa+pal+par+mcp | 0.005851 | 0.008061 | 2.904 | 4.738 |
| old_5task_a07 | bbh+pps+pal+par+mcp | 0.033814 | 0.046858 | 4.129 | 5.680 |
| old_5task_chain_bbh_a07 | bbh+pps+pal+par+mcp | 0.052359 | 0.075118 | 6.485 | 9.106 |

### 9.2.1 bbh/pal/par per-task KL

下表只展开 `Tasks = bbh+pal+par` 的候选。`bbh KL`、`pal KL`、`par KL`
分别是同一个 candidate 在对应 task observation 上的 `candidate@task KL/dim`。
这三列来自 `results/forward_kl_merge_sde20/forward_kl_rows.csv` 的 per-task rows；
`Avg/Max` 仍是上一张 summary 表中的聚合值。

| Candidate | Avg KL/dim | Max KL/dim | bbh KL | pal KL | par KL |
|---|---:|---:|---:|---:|---:|
| grid_s50_bbh05_pal05_par05 | 0.002293 | 0.003274 | 0.001843 | 0.003274 | 0.001763 |
| grid_s50_bbh07_pal05_par05 | 0.002604 | 0.003608 | 0.002200 | 0.003608 | 0.002005 |
| grid_s50_bbh05_pal07_par05 | 0.002890 | 0.003971 | 0.002434 | 0.003971 | 0.002264 |
| grid_s50_bbh09_pal05_par05 | 0.003044 | 0.004169 | 0.002539 | 0.004169 | 0.002423 |
| grid_s50_bbh07_pal07_par05 | 0.003217 | 0.004332 | 0.002770 | 0.004332 | 0.002549 |
| grid_s50_bbh09_pal07_par05 | 0.003811 | 0.005010 | 0.003297 | 0.005010 | 0.003126 |
| grid_s50_bbh05_pal05_par07 | 0.003458 | 0.005180 | 0.002594 | 0.005180 | 0.002602 |
| grid_s50_bbh07_pal05_par07 | 0.003707 | 0.005330 | 0.002968 | 0.005330 | 0.002823 |
| grid_s50_bbh05_pal07_par07 | 0.004060 | 0.005720 | 0.003272 | 0.005720 | 0.003188 |
| grid_s50_bbh09_pal05_par07 | 0.004133 | 0.005960 | 0.003327 | 0.005960 | 0.003112 |
| grid_s50_bbh07_pal07_par07 | 0.004433 | 0.006289 | 0.003604 | 0.006289 | 0.003405 |
| s50_bbh07_pal07_par07 | 0.004450 | 0.006321 | 0.003601 | 0.006321 | 0.003426 |
| grid_s50_bbh09_pal07_par07 | 0.004938 | 0.007009 | 0.004016 | 0.007009 | 0.003791 |
| s50_bbh09_pal07_par07 | 0.004938 | 0.007009 | 0.004016 | 0.007009 | 0.003791 |

### 9.3 SDE-20 更新后的 alpha 结论

SDE-20 后总体排序和单轨迹版一致，但数值更稳定，后续以 `results/forward_kl_merge_sde20` 为准。

`bbh/pal/par` step50 的低风险候选排序：

| Alpha | Avg KL/dim | Max KL/dim | Avg ratio | Max ratio | 判断 |
|---|---:|---:|---:|---:|---|
| bbh=0.5, pal=0.5, par=0.5 | 0.002293 | 0.003274 | 1.262 | 1.924 | 最保守，优先 eval |
| bbh=0.7, pal=0.5, par=0.5 | 0.002604 | 0.003608 | 1.439 | 2.120 | 次优，给 bbh 更多权重 |
| bbh=0.5, pal=0.7, par=0.5 | 0.002890 | 0.003971 | 1.592 | 2.334 | pal 优先时可 eval |
| bbh=0.9, pal=0.5, par=0.5 | 0.003044 | 0.004169 | 1.671 | 2.450 | bbh 优先的探索候选 |
| bbh=0.7, pal=0.7, par=0.7 | 0.004433 | 0.006289 | 2.442 | 3.696 | 已偏高，不应作为默认起点 |
| bbh=0.9, pal=0.7, par=0.7 | 0.004938 | 0.007009 | 2.721 | 4.119 | 高风险，只适合 bbh 单任务优先 |

更新后的建议：

1. 后续融合筛选默认使用 20 条 base Flow-SDE trajectory，而不是单条 base ODE/mean path。
2. `bbh/pal/par` 下一批 eval 顺序仍是：`0.5/0.5/0.5`、`0.7/0.5/0.5`、`0.5/0.7/0.5`。
3. `par alpha=0.7` 会明显推高 pal/bbh KL；如果 par 当前表现够强，先保持 `par=0.5`。
4. `0.7/0.7/0.7` 和 `0.9/0.7/0.7` 的结论不变：前者是中高风险，后者对 pal 的遗忘风险更高。
5. `old_5task` 和 `chain` 的 collapse 仍被 SDE-20 明确捕捉，chain 仍显著更差。

---

## 10. bbh/pal/par alpha grid 三任务成功率

### 10.1 评测设置

- 指标：`eval/success_once`
- 任务：`beat_block_hammer`、`place_a2b_left`、`place_a2b_right`
- 每个任务使用对应的 96 episode manifest，`eval_rollout_epoch: 3`
- 新 grid 权重生成脚本：`scripts/merge_bbh_pal_par_s50_grid.py`
- 评测脚本：`examples/embodiment/slurm/eval_bbh_pal_par_s50_grid.sbatch`
- 机器可读结果：`results/forward_kl_merge_sde20/grid_success_rates.csv`

Job 记录：

- `10209446`：完成前 8 个子任务，随后在 Ray collective 错误后卡死并取消
- `10209639`：恢复并完成第三至第五个候选，其中一个 pal eval timeout
- `10209640`：完成第六至第八个候选
- `10209641`：完成第九、第十个候选
- `10209727`：补跑 `grid_s50_bbh07_pal07_par05/place_a2b_left`
- `0.7/0.7/0.7` 和 `0.9/0.7/0.7` 复用同协议已有 jobs `10206142/10206391/10206403`

### 10.2 完整成功率

| Candidate | bbh | pal | par | bbh success | pal success | par success | Avg success | SDE-20 max ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| grid_s50_bbh05_pal05_par05 | 0.5 | 0.5 | 0.5 | 69.7% | 49.8% | 71.4% | 63.6% | 1.924 |
| grid_s50_bbh07_pal05_par05 | 0.7 | 0.5 | 0.5 | 72.5% | 65.5% | 60.8% | 66.3% | 2.120 |
| grid_s50_bbh05_pal07_par05 | 0.5 | 0.7 | 0.5 | 76.1% | 51.7% | 71.6% | 66.5% | 2.334 |
| grid_s50_bbh09_pal05_par05 | 0.9 | 0.5 | 0.5 | 60.4% | 55.4% | 62.8% | 59.6% | 2.450 |
| grid_s50_bbh07_pal07_par05 | 0.7 | 0.7 | 0.5 | 73.3% | 64.7% | 62.9% | 67.0% | 2.546 |
| grid_s50_bbh05_pal05_par07 | 0.5 | 0.5 | 0.7 | 75.1% | 64.1% | 66.9% | **68.7%** | 3.044 |
| grid_s50_bbh07_pal05_par07 | 0.7 | 0.5 | 0.7 | **78.6%** | 58.5% | 56.5% | 64.5% | 3.133 |
| grid_s50_bbh09_pal07_par05 | 0.9 | 0.7 | 0.5 | 74.9% | 60.5% | 63.1% | 66.2% | 2.945 |
| grid_s50_bbh05_pal07_par07 | 0.5 | 0.7 | 0.7 | 69.0% | 65.7% | 61.1% | 65.2% | 3.362 |
| grid_s50_bbh09_pal05_par07 | 0.9 | 0.5 | 0.7 | 67.1% | 68.8% | 67.2% | **67.7%** | 3.503 |
| grid_s50_bbh07_pal07_par07 | 0.7 | 0.7 | 0.7 | 60.9% | **71.0%** | 66.1% | 66.0% | 3.696 |
| grid_s50_bbh09_pal07_par07 | 0.9 | 0.7 | 0.7 | 73.7% | 55.5% | 62.3% | 63.8% | 4.119 |

### 10.3 结果判断

**最高平均成功率**：

- `bbh=0.5, pal=0.5, par=0.7`
- 平均成功率 68.7%
- 三任务分别为 75.1% / 64.1% / 66.9%

**最均衡候选**：

- `bbh=0.9, pal=0.5, par=0.7`
- 三任务分别为 67.1% / 68.8% / 67.2%
- 最低任务成功率仍有 67.1%，max-min spread 只有约 1.8pp
- 平均成功率 67.7%

**低 KL 与成功率的较好折中**：

- `bbh=0.7, pal=0.5, par=0.5`
- SDE-20 max ratio 2.120
- 三任务平均成功率 66.3%
- 相比 `0.5/0.5/0.5`，主要改善 pal，但 par 有所下降

**不推荐**：

- `bbh=0.9, pal=0.5, par=0.5` 平均只有 59.6%，说明单独提高 bbh alpha 并不能稳定提升 bbh 或整体性能。
- `bbh=0.9, pal=0.7, par=0.7` max ratio 4.119，平均成功率仅 63.8%，高 KL 风险与较差整体结果一致。

### 10.4 对 forward-KL 指标的修正认识

实验说明 forward-KL 更适合作为风险指标，而不是成功率排序指标：

- 最低 KL 的 `0.5/0.5/0.5` 并不是成功率最高候选，pal 只有 49.8%。
- 最高平均成功率 `0.5/0.5/0.7` 的 max ratio 已达到 3.044。
- 最均衡候选 `0.9/0.5/0.7` 的 max ratio 为 3.503，但三任务成功率都在 67% 左右。
- 高 KL 并不必然失败，但代表策略相对 solo/base 的分布偏移更大，需要真实 eval 验证。

因此 alpha 筛选建议更新为：

1. 不再把 `max ratio > 3` 当作硬过滤条件，而是高风险标记。
2. 先用 forward-KL 排除极端候选，再在中等与高风险候选中保留少量 reward 探索点。
3. 最终权重选择必须看三任务真实成功率，不能只按 KL 最小化。
4. 当前优先候选为：
   - 追求平均：`0.5/0.5/0.7`
   - 追求均衡：`0.9/0.5/0.7`
   - 追求较低 KL：`0.7/0.5/0.5`

### 10.5 方差注意事项

恢复脚本变量未穿透容器时，曾意外重复评测一次
`grid_s50_bbh05_pal05_par05/beat_block_hammer`：

- 正式表格采用的第一次结果：69.7%
- 意外重复结果：78.5%

这说明当前带采样的单次 eval 存在不可忽略的方差。上表适合筛选，但对 top candidate 的最终结论应至少再独立重复 2 次，并报告 mean/std。
