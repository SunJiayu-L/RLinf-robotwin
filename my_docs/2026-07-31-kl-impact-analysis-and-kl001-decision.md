# Reference-KL 对 handover_block 的实际影响：数据分析与 β=0.001 诊断实验

**日期：** 2026-07-31（最后更新 2026-08-05，含 3.7 节评测方差重跑结果）
**Branch：** `KL`
**关联文档：** [`2026-07-18-handover-block-ppo-kl-constraint.md`](2026-07-18-handover-block-ppo-kl-constraint.md)（实现细节）、[`2026-07-31-why-KL.md`](2026-07-31-why-KL.md)（设计动机）、[`2026-07-29-handover-block-active-experiments.md`](2026-07-29-handover-block-active-experiments.md)（实验总览）

---

## 一、背景

`2026-07-31-why-KL.md` 记录的设计动机：`handover_block` 标准 PPO（`kl_beta=0`，从 SFT 直接训练）曾多次训练失败，成功率从 SFT baseline 37.5% 降到 29.2%。为此引入 reference-policy KL 约束（`ref_kl_beta`，锚定 5-task 联合 PPO step-100 checkpoint），期望在保留 PPO 在线奖励信号的同时，避免破坏 SFT/5-task 学到的有效动作结构。

目前唯一实际施加了这个约束的活跃实验是**实验 B**（`handover-block-ppo200-kl01-fix2-4g-m800-0730-1218`，`ref_kl_beta=0.01`，`kl_penalty=low_var_kl`，从 `global_step_200` 续训）。本文档基于实验 A（同起点、`ref_kl_beta=0`）和实验 B 的真实 TensorBoard/`run_embodiment.log` 数据做定量分析。

---

## 二、数据来源与方法

- 从两边 `run_embodiment.log` 用正则解析 Metric Table（`Global Step`、`success_once`、`reward`、`actor/*`、`critic/*`）
- A、B 均从 `global_step_200` 出发，PPO 超参一致（`rollout_epoch=4`、`update_epoch=5`、`global_batch_size=2048`、`micro_batch_size=32`、128 train envs、4 GPU）
- 对比区间取两边共同覆盖的 **step 201–251（51 步，即 B 的完整可用训练区间，B 在 251 步后因 Ray OOM 停止）**

---

## 三、关键发现

### 3.1 success_once：A 组稳健上升，B 组停滞/下降

| | A（无 KL） | B（ref_kl_beta=0.01） |
|---|---|---|
| 均值（201–251） | 0.282（σ=0.067） | 0.241（σ=0.063） |
| 线性回归窗口内变化（201→251） | **+7.4pp**（持续上升） | **-3.9pp**（持平/略降） |
| Welch t 检验（均值差） | 差值 0.041，t=3.19（n=51 each） | 非纯噪声，是真实信号 |

### 3.2 ref_kl_loss：有下降，但幅度很小

对 B 的 51 个 step 做线性回归：

```
slope = -0.00745/step, SE=0.00309, t=-2.41（统计显著）
预测区间内下降：4.65 → 4.28，相对下降 ~8.3%
```

**结论修正**：KL 并非完全没有下降（早期简单看窗口均值容易误判为"持平"），但下降幅度很小——统计上真实，实践上几乎可忽略。

### 3.3 核心机制：KL 惩罚项量级 ≈ PPO 信号量级的一半

用 A 的 `actor/policy_loss` 均值（-0.0919）作为"纯 PPO 梯度信号量级"的参照（A/B 除 KL 外完全同超参、同起点，可互为对照）：

```
KL 惩罚项贡献 = ref_kl_beta × ref_kl_loss ≈ 0.01 × 4.46 ≈ 0.0446
纯 PPO 信号量级（A 的 policy_loss）      ≈ 0.0919
KL 项 / PPO 信号 ≈ 49%
```

反推 B 中"去掉 KL 项后的纯 PPO 部分" = -0.041 - 0.045 ≈ -0.086，与 A 的 -0.092 高度吻合，验证了这个换算的合理性。

**这解释了两个现象共存的原因**：β=0.01 时 KL 惩罚在 loss/梯度中的权重接近 PPO 奖励信号的一半——大到足以明显压制 success_once 提升，但由于 KL 惩罚梯度和 PPO 奖励梯度方向相反、彼此抗衡，又不够大到真正把策略拉回参考策略附近，导致 `ref_kl_loss` 只能在一个平台附近缓慢漂移（8%/50步），是"双输"的中间强度。

### 3.4 场景错配的可能性

`why-KL.md` 描述的崩溃案例（37.5%→29.2%）来自**从 SFT（step 0）直接无约束 PPO**的场景。而实验 B 是**从已经稳定训练到 step 200 的 checkpoint 续训**；同样从 step 200 续训但不加 KL 的实验 A/E，并未表现出崩溃迹象，反而在稳健提升。也就是说，当前测试 KL 的场景可能已经不是这个正则化设计原本要解决的问题——`moe_sft20k_handover_block_ppo_kl.yaml`（β=0.01，`resume_dir: null`，即从 SFT 直接开始）这个"原始设计场景"的配置在仓库里存在，但从未真正跑过对照实验。

### 3.5 β=0.001 验证结果（2026-08-04 更新）

实验 F（见第五节）跑到 step 254，已覆盖与 B 完全对齐的 51-step 窗口（202–251，剔除续训冷启动的 step 201，见下方"冷启动"说明）。三组在同一起点（`global_step_200`）、同一窗口下的对比：

| | A（无 KL） | **KL β=0.001（实验 F）** | B（KL β=0.01） |
|---|---|---|---|
| success_once 均值（202–251，n=50） | 0.282 | **0.255** | 0.241 |
| 趋势 | 明显上升（+7.4pp，t 显著） | **基本走平**（斜率 t=-0.19，不显著） | 明显下降（-3.9pp） |
| ref_kl_loss 趋势 | 未计算（未加载 ref policy） | **基本走平/略升**（202→254 几乎无变化） | 缓慢下降（~-8%/50步） |

**结论**：β 从 0.01 降到 0.001 后：

- "success_once 持续下降"的问题解决了——不再下滑，变成走平。
- 但也没有恢复到无 KL 那种健康上升趋势——均值仍比 A 低约 2.7pp（比 β=0.01 时的 4.1pp 差距有缩小，但没有消失）。
- `ref_kl_loss` 不再像 β=0.01 那样有统计显著的缓慢下降，基本走平甚至略升——说明这个强度已经弱到几乎不产生"往参考策略拉"的效果。

也就是说 β=0.01 和 β=0.001 是两个不理想的极端：前者力度大到既拖累任务又勉强有点拉回效果，后者拖累消失了但拉回效果也基本消失了。**中间值（例如 0.003~0.005）大概率是更合理的下一步 sweep 点**，如果目标是"既不明显拖累 success_once、又要让 KL 有实质性收缩"。

另外一个旁证：同期观察的实验 E（无 KL，从 `global_step_250` 续训，见第五节末尾）在其自身窗口（252–301，50 步）的 success_once 也基本走平（均值 0.292，t=-0.035 不显著），并不像 A 在 step 200–250 区间那样持续上升。这提示 A 观察到的上升趋势可能只是 step 200–250 这一段"补涨"，success_once 到 step ~250 附近本身就会自然趋于平台（约 0.29–0.30），与是否加 KL 无关。因此上面"β=0.001 均值比 A 低 2.7pp"这个差距，也可能部分是"对比窗口所处训练阶段不同"造成的，而不是纯粹的 KL 效应——**更严谨的做法是让 KL(0.001) 也跑到 step 250+ 后，直接对比同一 step 区间内 KL(0.001) vs 实验 E（两者都从各自续训点出发，且都已进入"自然平台期"），排除训练阶段本身的混杂因素**。

### 3.6 5-task 固定种子评测对比（step 300，2026-08-05）

上面 3.1–3.5 的对比都基于训练过程中 rollout 阶段的在线 `success_once`（小批量、随机种子，噪声大）。为了拿到更干净、更能跟 `why-KL.md` 里引用的 baseline 数字（SFT 37.5%）对齐口径的结果，对 KL(0.001)（实验 F）和无 KL（实验 E）在**同一个 step（300）**上分别跑了一次标准的 5-task 固定种子评测。

**评测方法**：每个任务（bbh/ca/cb/hb/mpa）用专门的 seed manifest 文件（`rlinf/envs/robotwin/seeds/<task>_manifest_96ep.json`，每个任务 96 条固定 `(seed, instruction)`），配合 `use_fixed_reset_state_ids: True`，env reset 时按 `partition_episode_manifest` 分片后顺序、不重复地消费。`env.eval.total_num_envs=32` × `algorithm.eval_rollout_epoch=3` = 96，理论上覆盖全部 96 条种子；但由于 `auto_reset: True` + `ignore_terminations: True`，episode 天然较短的任务会在同一个 400-step 窗口内完成并重置多次，实际消耗的 trajectory 数会超过 96（例如 bbh 跑了 325 条），episode 接近占满窗口的任务则精确等于 96（cb、hb）。这个机制细节还没有完全查清楚（manifest 严格来说是有限且不重复的，超出理论应报错，但实际没有报错），标记为待确认项。

新建了 10 个 eval config（`examples/embodiment/config/eval_kl001_s300_{bbh,ca,cb,hb,mpa}.yaml` 和 `eval_nokl_s300_{bbh,ca,cb,hb,mpa}.yaml`，从已有的 `eval_ppo5task_s200_*.yaml` 模板改造，替换成本地集群路径和对应 checkpoint），提交脚本 [`submit_eval_kl001_s300_5task_openpi_vulkan_4gpu.sh`](submit_eval_kl001_s300_5task_openpi_vulkan_4gpu.sh) / [`submit_eval_nokl_s300_5task_openpi_vulkan_4gpu.sh`](submit_eval_nokl_s300_5task_openpi_vulkan_4gpu.sh)，各用 4 GPU / 400GB 依次跑完 5 个任务。

**结果**：

| 任务 | KL β=0.001 | 无 KL | 差值（KL − 无KL） |
|---|---|---|---|
| bbh（beat_block_hammer） | 61.5% | 59.2% | +2.3pp |
| ca（click_alarmclock） | 75.0% | 84.1% | −9.1pp |
| cb（click_bell） | 82.3% | 81.3% | +1.0pp |
| **hb（handover_block）** | **62.5%** | **55.2%** | **+7.3pp** |
| mpa（move_playingcard_away） | 81.0% | 80.6% | +0.4pp |
| **均值** | **72.5%** | **72.1%** | **+0.4pp** |

**解读**：

- 五任务均值几乎持平（+0.4pp），说明 β=0.001 这个强度下，KL 约束**没有拖累**整体多任务表现。
- 在训练目标任务 handover_block 本身，KL 组明显更高（+7.3pp）——这跟 3.5 节用训练时 on-policy 指标做的对比（KL(0.001) 走平、没有超过无 KL 组）方向不完全一致，原因很可能是评测方式不同：训练时的 `success_once` 是小批量随机种子的在线读数噪声大，这里是固定种子、确定性评测，样本更干净，且是更晚的 step（300 vs 之前对比的 251 附近）。
- click_alarmclock 上 KL 组明显更差（-9.1pp），是唯一一个 KL 组落后超过 5pp 的任务，需要留意。
- **局限**：每个 config 只跑了一次，没有多组随机种子重复评测来估计方差，单任务 ±5~9pp 的差异有多少是真实效应、多少是评测噪声还不能完全确定。目前能下的结论是"β=0.001 至少没有对整体多任务能力造成净损害，且在目标任务上有正向信号"，还不能下"KL 约束确定性地帮助了 handover_block"这么强的结论。

**⚠️ 见 3.7 节：这个结论被后续重复评测推翻了。**

### 3.7 5-task 评测重跑（2026-08-05）：单次评测的方差比 KL vs 无KL 的差异还大

3.6 节的结论只基于一次评测。用完全相同的 checkpoint（KL(0.001) 和无 KL 都是 `global_step_300`）、完全相同的 eval config，重新各跑一遍 5-task 评测（`eval_kl001_s300_*` / `eval_nokl_s300_*` 原样重新提交）。由于 eval 配置里 `sampling_params.do_sample: True`、`temperature_eval: 0.6`，策略动作采样本身有随机性（环境侧种子是固定的，但策略输出不是确定性的），即使 checkpoint 和固定种子完全一样，重复跑的结果也会有波动。

**两次评测汇总**：

| 任务 | KL run1 | KL run2 | KL 均值 | 无KL run1 | 无KL run2 | 无KL 均值 | 均值差(KL−无KL) |
|---|---|---|---|---|---|---|---|
| bbh | 61.5% | 53.4% | 57.5% | 59.2% | 51.3% | 55.3% | +2.2pp |
| ca | 75.0% | 87.1% | 81.1% | 84.1% | 91.4% | 87.8% | −6.7pp |
| cb | 82.3% | 88.5% | 85.4% | 81.3% | 82.3% | 81.8% | +3.6pp |
| **hb** | 62.5% | 51.0% | **56.8%** | 55.2% | 52.1% | **53.7%** | **+3.1pp** |
| mpa | 81.0% | 85.5% | 83.3% | 80.6% | 88.7% | 84.7% | −1.4pp |
| **五任务均值** | 72.5% | 73.1% | **72.8%** | 72.1% | 73.2% | **72.6%** | **+0.2pp** |

**单次评测的波动幅度**（同一 checkpoint 两次评测的绝对差）：

- KL 组：bbh 8.1pp、ca **12.1pp**、cb 6.2pp、hb **11.5pp**、mpa 4.5pp
- 无KL 组：bbh 7.9pp、ca 7.3pp、cb 1.0pp、hb 3.1pp、mpa 8.1pp

**结论：3.6 节"KL 组在 handover_block 上 +7.3pp、click_alarmclock 上 -9.1pp"的结论不成立。** 单次评测的波动幅度（普遍 5~9pp，个别任务到 11~12pp）跟 3.6 节里观察到的 KL vs 无KL 差异是同一量级。两轮取均值后：

- hb 上 KL 只领先 3.1pp（run1 的 +7.3pp 里，一大半是噪声）
- ca 上 KL 反而落后 6.7pp（run1 显示落后 9.1pp，方向一致但幅度也在变）
- 五任务总均值两边几乎完全打平（72.8% vs 72.6%，只差 0.2pp）

**现在能确认的、比较站得住的结论只有**："β=0.001 的 KL 约束在 step 300 附近，对五任务整体表现没有造成可辨识的净损害"——仅此而已。"KL 约束帮助了 handover_block 本身"这个更强的说法，从目前两轮数据看，还不能确认，需要更多轮次评测（至少 4-5 轮）取更稳定的均值才能判断这类个位数到十位数百分点的差异是否是真实效应。

---

## 四、下一步决策

优先做成本最低、最直接的诊断：**把 β 从 0.01 降到 0.001，其余条件（起点、超参、参考策略）完全不变**，验证"HB 学不动"是否纯粹是 KL 力度过大导致。

按上面的量级换算，β=0.001 时 KL 项贡献 ≈ 0.001×4.46 ≈ 0.0045，约为 PPO 信号量级的 5%，理论上不应再明显压制 success_once 上升趋势。

- 仓库中已有配置 `examples/embodiment/config/moe_sft20k_handover_block_ppo_kl001_fromhb200.yaml`（β=0.001，从 step 200 续训），此前从未提交运行过。
- 已有 slurm sbatch（`examples/embodiment/slurm/moe_sft20k_handover_block_ppo_kl001_fromhb200.sbatch`），但路径指向 `/mnt/petrelfs/...`（另一集群），不能直接在当前 rjob 环境用；本次改为新写一个 rjob 提交脚本，复用实验 B 脚本（`submit_handover_block_ppo200resume_kl01_openpi_vulkan_4gpu.sh`）的结构，仅将 `ref_kl_beta` 改为 `0.001`。
- 鉴于目前所有加了 KL 的续训实验（B、以及无 KL 的实验 C）都撞上过 Ray OOM，本次额外加上实验 D/E 验证过的 no-video/no-image 设置（`env.train.video_cfg.save_video=False` 等），提高跑到有意义 step 数的概率，避免像 B 一样只拿到 51 步数据。

**判断标准**（后续对比时用）：
- 若 β=0.001 下 success_once 恢复接近 A 的上升趋势 → 验证是"力度问题"，可在 0.001~0.01 之间做小范围 sweep 找平衡点。
- 若 β=0.001 依然明显压制 → 说明问题不只是系数大小，需要重新审视参考策略选择（5-task step-100 是否合适）/ KL 估计器 / 加入位置本身。
- 无论哪种结果，仍需要补一次"从 SFT（step 0）+ KL β=0.01”对照，来验证这个方法在它本来设计要解决的场景下是否真的有效（当前完全没有这组数据）。

**实际结果（见 3.5）**：介于两者之间——不再压制，但也没有恢复上升趋势，KL 本身也基本不收缩了。

**已完成**：第 2 项（同阶段对比）已经用固定种子的 5-task 评测做完，见 3.6 节，并在 3.7 节用第二轮重复评测验证了方差——两轮取均值后，五任务总均值基本打平（+0.2pp），单任务差异（包括之前看起来很显著的 hb +7.3pp）大多落在评测本身的采样噪声范围内。

**仍然待办**：
1. 3.7 节已经证明"跑一次评测就下结论"不可靠，如果还想验证 β=0.001 对 handover_block 是否有真实（哪怕很小）的正面效应，需要至少 4-5 轮重复评测取更稳定的均值，或者改用更大的 eval 样本量 / 关闭采样随机性（`do_sample: False`）来降低评测本身的方差；
2. 在 0.003~0.005 之间取一个中间 β 值做 sweep 目前优先级降低——既然 β=0.001 相对无 KL 看不出明显差异（正面或负面都在噪声范围内），先把评测方法的方差问题解决，再决定要不要继续 sweep 更多 β 值；
3. "从 SFT（step 0）+ KL β=0.01"这组原始设计场景的对照实验仍未补，如果要验证 KL 约束在它本来要解决的问题（SFT 直接 PPO 崩溃）上是否有效，这组数据依然缺失。

---

## 五、实验 F：step-200 + 5-task 参考 KL（0.001）

| 项目 | 设置 |
|---|---|
| 提交时间 | 2026-07-31（本文档提交） |
| 提交脚本 | [`submit_handover_block_ppo200resume_kl001_openpi_vulkan_4gpu.sh`](submit_handover_block_ppo200resume_kl001_openpi_vulkan_4gpu.sh) |
| 起点 | 与实验 A/B 相同的 `global_step_200` DCP checkpoint |
| 参考策略 | 5-task joint PPO step-100（同实验 B） |
| KL 参数 | `kl_penalty=low_var_kl`、`ref_kl_beta=0.001` |
| 最大 step | 500 |
| image/video | 关闭（沿用实验 E 的 no-video/no-image 设置，缓解 128-env 长时间 rollout 的 OOM 风险） |
| 目的 | 验证 β=0.01→0.001 后 KL 惩罚量级降到 PPO 信号的 ~5%，success_once 提升是否恢复 |

**第一次提交**（2026-07-31 18:02 HKT）：rjob `handover-block-ppo200-kl001-novideo-4g-m800-45365`（`--memory=800000`），跑到 step 211 时，因平台系统升级被 `Stopped`。

**重新提交**（2026-08-03 11:55 HKT，系统升级后）：内存从 800000 提到 1400000（`m1400`），其余配置不变；提交脚本同时把 gpfs2 checkpoint 的 host 侧 pre-flight check 移到了 rjob 容器内部（见脚本注释——升级后开发机本机的 gpfs2 挂载路径从 `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs` 变成了 `/mnt/ailab-eailabagent-gpfs`，本机侧检查可能因为挂载点变化而误报；容器内的 `--mount` 目标路径未变，不受影响，训练本身正常）。

| 项目 | 设置 |
|---|---|
| rjob | `handover-block-ppo200-kl001-novideo-4g-m1400-2f3f1` |
| showname | `handover-block-ppo200-kl001-novideo-4g-m1400-0803-1155` |
| 日志/新 checkpoint | `/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo200-kl001-novideo-4g-m1400-0803-1155`（本机侧读取用新挂载路径；容器内部路径仍是 `/mnt/shared-storage-gpfs2/...`） |
| 最终状态 | 人工在 2026-08-05 12:41 HKT 停止（`rjob stop`），最终停在 step 304/500；已保存 checkpoint `global_step_250`、`global_step_300` |

训练时 on-policy 指标对比结果见第 3.5 节；用 `global_step_300` checkpoint 做的固定种子 5-task 评测结果见第 3.6 节。

## 六、实验 E 重提交（无 KL 对照，用于排除训练阶段混杂因素）

实验 E（`handover-block-ppo250resume-novideo-4g-m800-e61fc`）同样在系统升级期间挂掉（状态 `Failed`）。重新提交后：

| 项目 | 设置 |
|---|---|
| rjob | `handover-block-ppo250resume-novideo-4g-m1400-058d8` |
| showname | `handover-block-ppo250resume-novideo-4g-m1400-0803-1155` |
| 内存 | 1400000（同样从 800000 提高） |
| 起点 | 实验 A 产出的 `global_step_250`（与原实验 E 相同） |
| 日志 | `/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo250resume-novideo-4g-m1400-0803-1155` |
| 状态（截至 2026-08-05 14:49 HKT） | `Running`，step 400/500，已保存 checkpoint `global_step_300`、`global_step_350`、`global_step_400` |

训练时 on-policy 指标用于第 3.5 节里"排除训练阶段混杂因素"的旁证对比（E 自身在 252–301 窗口 success_once 也基本走平，均值 0.292）；用 `global_step_300` checkpoint 做的固定种子 5-task 评测结果见第 3.6 节。
