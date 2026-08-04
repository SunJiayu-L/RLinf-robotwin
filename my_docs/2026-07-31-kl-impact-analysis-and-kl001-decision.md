# Reference-KL 对 handover_block 的实际影响：数据分析与 β=0.001 诊断实验

**日期：** 2026-07-31
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

**实际结果（见 3.5）**：介于两者之间——不再压制，但也没有恢复上升趋势，KL 本身也基本不收缩了。下一步建议：
1. 在 0.003~0.005 之间取一个中间 β 值做 sweep，寻找"success_once 不明显受压 + ref_kl_loss 有实质收缩"的平衡点；
2. 让 KL(0.001)（实验 F）和无 KL 的实验 E 都跑到同一个更靠后的 step 区间（例如都到 step 300+），做同阶段对比，排除"训练阶段本身处于自然平台期"这个混杂因素；
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
| 状态（截至 2026-08-04 14:15 HKT） | `Running`，step 254/500 |

对比结果见第 3.5 节。

## 六、实验 E 重提交（无 KL 对照，用于排除训练阶段混杂因素）

实验 E（`handover-block-ppo250resume-novideo-4g-m800-e61fc`）同样在系统升级期间挂掉（状态 `Failed`）。重新提交后：

| 项目 | 设置 |
|---|---|
| rjob | `handover-block-ppo250resume-novideo-4g-m1400-058d8` |
| showname | `handover-block-ppo250resume-novideo-4g-m1400-0803-1155` |
| 内存 | 1400000（同样从 800000 提高） |
| 起点 | 实验 A 产出的 `global_step_250`（与原实验 E 相同） |
| 日志 | `/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo250resume-novideo-4g-m1400-0803-1155` |
| 状态（截至 2026-08-04 14:15 HKT） | `Running`，step 328/500 |

这组数据目前主要用于第 3.5 节里"排除训练阶段混杂因素"的旁证对比（E 自身在 252–301 窗口 success_once 也基本走平，均值 0.292）。
