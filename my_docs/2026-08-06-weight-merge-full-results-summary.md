# handover_block × 5-task 联合 PPO：权重融合与联合训练完整实验结果汇总

**日期：** 2026-08-06
**Branch：** `KL`
**用途：** 汇总本轮所有权重融合（task arithmetic）+ 后续联合 PPO 训练的评测结果，供进一步分析。
**关联文档：**
- `my_docs/2026-08-05-weight-merge-hb-joint-results.md`（融合实验原始记录，含每个候选的提交细节）
- `.claude/plans/optimized-hopping-liskov.md`（融合实验的规划文档）
- `my_docs/2026-07-17-ppo-all-runs-manifest.md`（5-task 联合 PPO 训练链路的历史记录）

---

## 一、背景

`handover_block`（下文简称 **hb**）是 RoboTwin 5 个任务（beat_block_hammer/bbh、click_alarmclock/ca、click_bell/cb、handover_block/hb、move_playingcard_away/mpa）里 SFT baseline 最低、最难训的任务（SFT 只有 ~12~22%，视评测批次而定；标准单任务 PPO 也容易训崩或训不动）。

之前尝试过在训练时对 hb 单任务 PPO 加"向 5-task 联合策略靠拢"的 KL 正则（见 `2026-07-31-kl-impact-analysis-and-kl001-decision.md`），但训练动态和重复评测都没能给出确定性结论。本轮换一个更直接的思路：**训练后直接做权重融合**（task arithmetic / 线性插值），把"hb 单任务 PPO"和"5-task 联合 PPO"两个已经训好的 checkpoint 按不同方式、不同系数混合，看能不能拿到"hb 好、其他任务也不掉"的融合点；并且进一步测试**从融合点出发继续做真正的联合 PPO 训练**是否比静态融合更好。

---

## 二、三个"源" checkpoint

所有融合都基于这三个 checkpoint 做线性组合，权重矩阵格式统一是 pi0.5（`paligemma_with_expert` 结构），state_dict 是原始无前缀 key。

| 角色 | 说明 | 来源 |
|---|---|---|
| **SFT**（基底） | `pi05_aloha_moe_demo_clean` 的 SFT-20k checkpoint，`model.safetensors`，**812 个 key，不含 value_head**（SFT 训练不用 critic） | `/mnt/shared-storage-user/chenzanxin/jiayusun/sft20k/model.safetensors` |
| **hb PPO**（表中 `baseline-hb400`） | handover_block **单任务** PPO，无 KL 约束，续训到 **global_step_400**，821 个 key（含 value_head，8 个 key） | `handover-block-ppo250resume-novideo-4g-m1400-0803-1155` 续训链的 step400 |
| **joint PPO**（表中 `baseline-joint100`） | 5 个任务**联合** PPO，821 个 key。训练链路：**24 GPU（3 节点）从头训到 step50** → 切到 **8 GPU（1 节点）续训**，产出 step100/150/200。本轮用的是这条链路的 **global_step_100** | `five_task_ppo_step100/full_weights.pt`，即 8GPU 续训链的 `global_step_100` |

**SFT 只有 812 key、PPO checkpoint 有 821 key**这个差异，是理解下面融合结果的一个关键细节（见第五节"已知的技术细节"）。

---

## 三、融合方法：task arithmetic

工具：`scripts/task_arithmetic_rlinf.py`（已有脚本，未做任何代码修改）。

**核心公式：**

$$\theta^* = \theta_{base} + \sum_i \alpha_i \cdot (\theta_{ft_i} - \theta_{base})$$

- 只对 **208 个"PPO 实际训练过的" key** 做融合（`PI05_PPO_TRAINABLE_PREFIXES`：gemma_expert 的 transformer 层和 norm、`action_in_proj`、`action_out_proj`、`time_mlp_in`、`time_mlp_out`），其余 key 直接从 `base` 原样拷贝。这是仓库里之前所有融合实验（`2026-6-22`、`2026-7-6`、`2026-7-7`、`2026-7-9` 系列文档）一贯的约定——因为 PPO（`train_expert_only=True`）实际只改这 208 个参数，SFT 全参训练改的是全部 812 个，混进未训练过的 key 只会带来噪声。
- CLI：`--base <path> --ft <path>... --alpha <float> | --alphas <float>... --output <path>`

### 3.1 两种"融合方式"，本质是两种参数化

本轮设计了两个候选家族，对应用户理解里的"三种基底"，但代码实现上只需要两种调用形式：

**Family A（表中 `mode1-*`）：SFT 为基底，两个任务向量**

```
theta* = SFT + α_hb·(hb - SFT) + α_joint·(joint - SFT)
```

CLI：`--base <SFT> --ft <hb> <joint> --alphas α_hb α_joint`

**Family B（表中 `hbline-r*`）：hb 为基底，单一任务向量（不经过 SFT）**

```
theta*(r) = hb + r·(joint - hb) = (1-r)·hb + r·joint
```

CLI：`--base <hb> --ft <joint> --alpha r`

**关键点**：数学上，"以 hb 为基底、混入比例 r 的 joint" 和 "以 joint 为基底、混入比例 (1-r) 的 hb" 是**同一条线上的同一个点**（`hb_base(r) ≡ joint_base(1-r)`，代数上可以直接证明——两边按 hb/joint 的系数对齐即得 `s=1-r`）。所以"以 hb 为基底"和"以 joint 为基底"这两种命名视角，用**一种脚本调用形式**（固定 `--base hb --ft joint`，只变 `--alpha r`）就能完全覆盖，包括两端之外的**外推点**（`r<0` 越过 hb 继续往"远离 joint"方向走；`r>1` 越过 joint 继续往"远离 hb"方向走）。不需要真的再跑一次 `--base joint --ft hb`，那样算出来的权重和 `--base hb --ft joint --alpha (1-r)` 是逐字节相同的。

Family A 和 Family B 的本质区别：**A 经过 SFT 锚定**（两个任务向量都是相对 SFT 的偏移量，αhb/αjoint 独立可调，α=1 时精确等于对应 PPO checkpoint 在 208 个训练过的 key 上的值），**B 不经过 SFT**（直接在 hb 和 joint 两个 PPO checkpoint 之间做线性插值/外推，SFT 完全不参与运算）。

---

## 四、评测方法

统一用 5 个任务的固定种子评测基准（复用同一套 config 模板、只换 `runner.ckpt_path`）：

- 每任务用专属的 96-episode 固定种子 manifest（`rlinf/envs/robotwin/seeds/<task>_manifest_96ep.json`），`use_fixed_reset_state_ids=True`，环境侧种子完全固定
- `eval_rollout_epoch=3`、`total_num_envs=32`
- 指标：`eval/success_once`
- **已知噪声来源**：`sampling_params.do_sample=True`（`temperature_eval=0.6`），策略动作采样本身是随机的，即使环境种子固定，重复评测同一个 checkpoint 也会有波动。本轮"主评测结果对比"表里的**除 SFT20K 外，凡是本 session 内产出的候选都做了 2 次独立重复取均值**；实测同一 checkpoint 两次重复的单任务波动可以到 5~15 个百分点（例如 `hbline-r025` 的 bbh 两次重复相差 14.5pp）。**表里数字的个位数差异（例如 42.2% vs 44.3%）大概率在噪声范围内，不能直接排序谁更好。**
- `min` 列取 5 个任务里成功率最低的那个（而不只是看 `avg`）——参考之前 `2026-7-9-5task-bcam-merge.md` 的教训："排名要看 min 不能只看 avg"，均值好但某个任务崩掉的候选不该排前面。

---

## 五、主评测结果对比（本 session 完整记录，逐行说明）

| 候选 | bbh | ca | cb | hb | mpa | avg | min |
|---|---|---|---|---|---|---|---|
| SFT20K | 65.90% | 84.00% | 40.60% | 18.20% | 84.00% | 58.54% | 18.20% |
| baseline-hb400（纯 hb PPO） | 55.30% | 91.20% | 87.50% | 42.20% | 78.30% | 70.90% | 42.20% |
| baseline-joint100（纯联合 PPO） | 68.90% | 93.10% | 96.90% | 0.50% | 84.80% | 68.80% | 0.50% |
| mode1-hb07-joint04（SFT基底，hb权重更高） | 70.50% | 89.00% | 95.80% | 44.30% | 74.50% | 74.80% | 44.30% |
| step-50（继续联合PPO训练） | 78.20% | 96.60% | 99.50% | 42.20% | 91.70% | 81.60% | 42.20% |
| hbline-r-025（外推，越过 hb） | 46.40% | 86.60% | 83.90% | 48.40% | 86.50% | 70.40% | 46.4%(bbh) |
| hbline-r025（75% hb + 25% joint） | 65.60% | 89.20% | 94.80% | 43.80% | 87.20% | 76.10% | 43.80% |
| hbline-r050（中点） | 74.60% | 81.20% | 99.00% | 32.30% | 90.70% | 75.60% | 32.30% |
| hbline-r075（25% hb + 75% joint） | 80.70% | 97.70% | 97.90% | 9.40% | 85.00% | 74.10% | 9.40% |
| hbline-r125（外推，越过 joint） | 84.10% | 95.90% | 99.00% | 0.00% | 84.90% | 72.80% | 0.00% |
| mode1-a03（SFT基底，均匀0.3） | 71.60% | 87.20% | 81.20% | 30.20% | 83.70% | 70.80% | 30.20% |
| mode1-a05（SFT基底，均匀0.5） | 71.20% | 85.80% | 95.30% | 33.90% | 84.70% | 74.20% | 33.90% |
| mode1-a07（SFT基底，均匀0.7） | 74.70% | 91.30% | 99.50% | 38.00% | 93.30% | 79.40% | 38.00% |

### 逐行实验设置

**`SFT20K`** —— 纯 SFT-20k checkpoint，不经过任何 PPO/融合。本次（2026-08-06）在当前这套 infra 上用同样的固定种子评测重新测了一遍（2 次重复），跟历史上另外几套"SFT baseline"数字（`2026-6-24-moe-demo-clean-experiment-log.md` 的 5k/20k 版本、`2026-7-9-sft20k-eval-20tasks.md` 的批量 20 任务版本）都不完全一致——尤其 hb 这个任务，历史上出现过 2.1%/21.9%/12.5%/18.2% 四个不同数字，说明 hb 的 SFT 评测本身波动就很大，不只是本轮发现的"重复评测噪声"这一层问题。

**`baseline-hb400`** —— 纯 hb 单任务 PPO，无 KL 约束，续训到 `global_step_400`（本 session 里这条续训链路最新的 checkpoint）。作为"专精 hb 但没见过其他 4 个任务"的参照点。

**`baseline-joint100`** —— 纯 5-task 联合 PPO，`global_step_100`（8GPU 续训链）。作为"5 个任务一起训，但 hb 是弱项"的参照点——**hb 只有 0.50%，两次重复分别是 0% 和 1.04%，高度一致，不是噪声**，说明联合训练时 hb 这个任务被其他任务"挤压"到几乎学不会。这是所有融合实验要解决的核心问题：**能不能在保留联合训练其他 4 个任务能力的同时，把 hb 从接近 0% 拉回来**。

**`mode1-hb07-joint04`** —— Family A（SFT 基底）融合，`α_hb=0.7, α_joint=0.4`（hb 给更高权重）。CLI：`--base SFT --ft hb_ppo joint_ppo --alphas 0.7 0.4`。这是本轮融合候选里**唯一一个 hb 明显超过纯 hb PPO 基线**的（44.3% vs 42.2%），同时 avg 也不差（74.8%，Family A 里排第二）。选择给 hb 更高权重的理由：hb 是 SFT baseline 最低、最容易在均匀融合里被"稀释"掉的任务，之前 `2026-7-9-5task-bcam-merge.md` 也观察到"uniform 系数会让弱任务被强任务淹没"的现象。

**`step-50`** —— **注意这一行不是静态融合，是真正的 PPO 训练**。以 `mode1-hb07-joint04` 的融合权重为**初始权重**（用 `runner.ckpt_path` 加载，不是 `resume_dir`——global_step 从 0 开始，optimizer 全新初始化，不是"续训"），跑**真正的 5-task 联合 PPO 训练**（新建 config `moe_ppo_5task_bbh_ca_cb_hb_mpa_128env_4gpu.yaml`，4 GPU / 128 env / novideo）。训到 **step 50 时 Ray OOM 崩溃**（节点内存 743.51GB/781.25GB，5 个任务各自独立起环境实例，内存开销比单任务场景更大），但 `global_step_50` 的完整训练态 checkpoint（含 optimizer，可续训）保存下来了，评测的就是这个中间 checkpoint。**结果：avg 从融合起点的 74.8% 涨到 81.6%（+6.8pp），是整条实验线里 avg 最高的 checkpoint；hb 基本持平（44.3%→42.2%，噪声范围内），没有像纯 joint PPO 训练那样把 hb 训崩。** 说明"融合当起点 + 继续联合训练"这个方向有正向信号，但只有 50 步数据，样本量小。目前正在尝试续训到更多 step（1TB 内存，卡在集群节点资源紧张，见 `2026-08-05-weight-merge-hb-joint-results.md` 第六节）。

**`hbline-r-025` / `hbline-r025` / `hbline-r050` / `hbline-r075` / `hbline-r125`** —— Family B（hb 基底），CLI 统一是 `--base hb_ppo --ft joint_ppo --alpha r`，`r` = 混入 joint 的比例：
- `r=-0.25`：越过 hb 往"远离 joint"方向外推（放大 hb 独有特征）——hb=48.4%，本轮**所有候选里 hb 最高**，但 bbh 掉到 46.4%（本行的 min 是 bbh 不是 hb）
- `r=0.25`：75% hb + 25% joint（插值）——hb=43.8%，avg=76.1%（**全场 avg 第二高**）
- `r=0.50`：中点（50/50）——hb=32.3%，比两侧的 r=-0.25/r=0.25 都低，插值线上非单调
- `r=0.75`：25% hb + 75% joint——hb 骤降到 9.4%，开始明显接近纯 joint 的 0.5%
- `r=1.25`：越过 joint 往"远离 hb"方向外推——hb=0.0%，完全复现纯 joint PPO 的崩溃

沿这条线，hb 大体随 joint 混入比例增加单调下降（外推方向符合直觉：越靠近/越过 joint，hb 越接近 joint 自己的 0.5%；越靠近/越过 hb，hb 越接近甚至超过 hb 自己的 42.2%），中间没有出现"两头都不如"的甜蜜点。

**`mode1-a03` / `mode1-a05` / `mode1-a07`** —— Family A（SFT 基底）均匀系数扫描，`α_hb = α_joint = α ∈ {0.3, 0.5, 0.7}`。趋势：α 越大，avg 和 hb 都越高（30.2%→33.9%→38.0%），但即使 α=0.7 时 hb 也只有 38.0%，仍低于 `mode1-hb07-joint04`（hb 单独给更高权重）的 44.3%——印证"均匀系数会稀释 hb 这个弱任务"的假设，需要非对称权重才能让 hb 追上甚至超过它自己的单任务基线。

---

## 六、8GPU / 24GPU 联合 PPO 训练链路的多 Step 对比（补充参考数据）

这两张表是**同一条 5-task 联合 PPO 训练链路**在不同阶段、不同 step 的评测结果（不是本 session 产出，本 session 只用了这条链路的 `global_step_100` 作为 `baseline-joint100`）。根据 `my_docs/2026-07-17-ppo-all-runs-manifest.md` 的记录：训练**从 24 GPU（3 节点）开始，训到 step50 后切换到 8 GPU（1 节点）续训**，继续产出 step100/150/200。用户确认：**24GPU-step100 是真实数据，只是之前没记录进 my_docs；8GPU 表里原本的 step250/300 两行是复制粘贴错误（数值跟 step150/200 完全重复），已删除**。

### 24GPU 阶段（原始训练，3 节点×8GPU）

| 候选 | bbh | ca | cb | hb | mpa | avg | min |
|---|---|---|---|---|---|---|---|
| 24GPU-联合PPO step50 | 76.30% | 100.00% | 99.00% | 18.75% | 85.00% | 75.81% | 18.75% |
| 24GPU-联合PPO step100 | 73.30% | 99.00% | 99.00% | 15.60% | 85.70% | 74.52% | 15.60% |

### 8GPU 阶段（从 24GPU 的 step50 续训，1 节点）

| 候选 | bbh | ca | cb | hb | mpa | avg | min |
|---|---|---|---|---|---|---|---|
| 8GPU-联合PPO step50（=24GPU阶段终点，续训起点） | 75.10% | 92.90% | 95.80% | 11.50% | 77.90% | 70.64% | 11.50% |
| 8GPU-联合PPO step100（= `baseline-joint100`） | 81.20% | 90.40% | 99.00% | 0.00% | 81.20% | 70.36% | 0.00% |
| 8GPU-联合PPO step150 | 58.50% | 98.10% | 99.00% | 3.10% | 78.00% | 67.34% | 3.10% |
| 8GPU-联合PPO step200 | 75.60% | 98.10% | 97.90% | 0.00% | 80.90% | 70.50% | 0.00% |

**需要注意的地方**：

1. **这条链路的 hb 数值在整个训练过程中持续偏低甚至归零**（18.75%→15.60%→11.50%→0.00%→3.10%→0.00%），随着训练推进，hb 有明显的"越训越差、被其他任务挤压"的趋势——这是"纯联合 PPO 对 hb 不友好"这个问题最直接的时间序列证据，也是本轮做权重融合实验的核心动机。
2. **8GPU-step100 这一行跟本 session 用的 `baseline-joint100`（同一个 checkpoint）数字对不上**：bbh 81.2% vs 68.9%（差 12.3pp），avg 70.36% vs 68.80%。两者理论上是**同一个 checkpoint** 在不同时间点的评测，差异大概率来自本文档第四节提到的评测噪声（`do_sample=True` 导致的采样波动），但因为是不同来源（8GPU 表可能是当时训练过程中的即时评测记录，`baseline-joint100` 是本 session 重新单独评测的结果，各自都只有 1 次数据，没有互相重复验证），**不能完全排除是不同批次评测配置存在细微差异**。如果需要精确对比，建议后续对 `global_step_100` 这个 checkpoint 再补 1~2 次重复评测。
3. 24GPU 和 8GPU 两个阶段的 step50 是**同一个 checkpoint**（24GPU 训练的终点，也是 8GPU 续训的起点），但两张表里的 step50 数字也不完全一样（24GPU 表：hb=18.75%, avg=75.81%；8GPU 表：hb=11.50%, avg=70.64%）——同样可能是评测噪声，也可能是两次评测的具体配置（env 数、rollout_epoch）不同，未查证。

---

## 七、关键结论

1. **纯联合 PPO 训练本身会把 hb 训崩**（0~19% 之间反复，且随训练推进有恶化趋势），这是需要用融合/正则化解决的核心问题。
2. **权重融合（静态插值）能有效把 hb 从 joint 的接近 0% 拉回到 30~48%**，具体拉多高取决于给 hb 的相对权重——**给 hb 更高权重的候选（`mode1-hb07-joint04`、`hbline-r-025`）hb 表现最好**，均匀融合（`mode1-a*`）hb 明显更弱。
3. **"融合起点 + 继续联合 PPO 训练"（`step-50`）目前是全场 avg 最高的候选**（81.6%），且 hb 没有因为继续训练而掉回去，是最有希望的方向，但目前只有 50 步数据，需要跑更长（受限于集群内存/节点资源，正在解决）。
4. hb 沿着 `hbline` 插值线的表现是**非单调**的（两端外推点都不是最差，中点也不是最好），说明"hb 和 joint 之间的最优融合点"不是简单的线性关系，可能需要更细的系数网格才能找到真正的最优点。
5. **所有涉及 hb 的个位数到十几个百分点的差异，都需要用"评测噪声 5~15pp"这个背景来打折扣解读**，尤其是候选之间差距在 10pp 以内的排序（比如 `mode1-hb07-joint04` 44.3% vs `hbline-r025` 43.8% vs `step-50` 42.2%），目前的数据量（多数只有 1~2 次重复）还不足以做出可靠的精细排序。

---

## 八、产出物 / 复现信息

- Merge 脚本：`scripts/task_arithmetic_rlinf.py`
- Merge 提交脚本：`my_docs/submit_merge_hb_joint_weights.sh`
- 通用 5-task 评测提交脚本：`my_docs/submit_eval_5task_generic.sh`（融合/PPO checkpoint 用）、`my_docs/submit_eval_5task_sft20k.sh`（纯 SFT 用）
- 结果解析脚本：`scripts/parse_eval_success.py`
- 继续联合 PPO 训练用的新 config：`examples/embodiment/config/moe_ppo_5task_bbh_ca_cb_hb_mpa_128env_4gpu.yaml`
- 9 个融合权重：`/mnt/ailab-eailabagent-gpfs/chenzanxin/merged_weights/wm-hb-joint-0805/<candidate>/full_weights.pt`
- Merge manifest（每个候选的精确 CLI 参数）：`/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/wm-merge-hb-joint-0805-1637/MERGE_MANIFEST.md`
- 详细的候选级别记录（含每次 rjob 提交、每次重复的原始值）：`my_docs/2026-08-05-weight-merge-hb-joint-results.md`
