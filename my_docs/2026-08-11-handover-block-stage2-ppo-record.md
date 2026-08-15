# handover_block Stage-2 PPO 实验记录

**日期：** 2026-08-11  
**Branch：** `KL`  
**状态：** Stage2 已完成；Stage3 正从 Stage2 step 125 继续训练  
**目标：** 使用不超过 8 GPU 的单任务 PPO，将 `handover_block` 固定评测成功率提升到约 90%。

## 一、结论摘要

- 旧续训配置存在首轮 rollout 权重未与 actor checkpoint 全量同步的问题。旧实验每次 resume 的首个 `approx_kl` 为 1.75 到 14.00，正常训练仅约 0.01 到 0.03。本轮通过 `weight_syncer.patch.init_sync.enabled=true` 修复。
- 256 train env 在单个 H200 节点上不可持续。运行到 stage2 step 27 时，8 个 EnvWorker 合计占用约 1359 GB，节点总内存越过 Ray 95% 阈值后 OOM。
- 从 stage2 step 25 改为 128 train env 后，训练稳定运行到 step 150，未再出现 OOM。
- 2026-08-12 复查代码和 TensorBoard 后确认：只有每个新进程的第一次评测覆盖 96 个 manifest episode；同一进程的后续评测因 manifest 游标未回卷而失效。Stage2 step 75/100/125/150 的 `num_trajectories=544`，实际反复统计最后 32 个 episode，不能按 96 条解释。
- OpenPI rollout 不使用 `temperature_eval` 参数。eval 虽然采用 ODE 去噪，但每个 action chunk 仍重新采样初始高斯噪声，因此不是确定性评测。
- Stage2 step 125 的 78.1% 应解释为约 `25/32` 的一次随机结果，不是 `75/96`；step 150 的 59.4% 约为 `19/32`。两者同时包含策略变化和较大的评测方差，当前无法据此认定真实下降 18.75pp。
- Stage3 继续运行；其第一次周期评测（step 25）会在新进程中覆盖 96 条，但仍有策略采样随机性。后续 checkpoint 需要独立新进程复评，或先修复 manifest 回卷和 eval RNG。

## 重要评测更正（2026-08-12）

TensorBoard 给出了直接证据：

| checkpoint | 日志中的 success | `num_trajectories` | `episode_len` | 正确解释 |
|---:|---:|---:|---:|---|
| step 50 | 55.21% | 96 | 400 | 53/96，首次评测有效 |
| step 75 | 59.38% | 544 | 800 | manifest 已耗尽；约 19/32 被重复统计 |
| step 100 | 62.50% | 544 | 800 | manifest 已耗尽；约 20/32 被重复统计 |
| step 125 | 78.13% | 544 | 800 | manifest 已耗尽；约 25/32 被重复统计 |
| step 150 | 59.38% | 544 | 800 | manifest 已耗尽；约 19/32 被重复统计 |

原因链路：`EnvWorker.evaluate()` 每次评测只调用普通 `reset()`，但没有重置 `_current_episode_index`；manifest 首轮耗尽后，`RoboTwinEnv._handle_auto_reset()` 在越界时直接返回且不 reset。此后 truncation 持续为真，同一批最后 32 个环境在剩余 chunk 中被重复加入指标，形成 `32 * 17 = 544` 条记录和平均 `episode_len=800`。

因此，下文早期记录中所有“step 75 之后固定 96 条、确定性评测”的措辞均由本节更正；原始数值保留用于追溯，但不再作为 90% 目标的有效判据。

## 二、起点 checkpoint

本轮使用已验证表现最好的 `handover_block` KL=0.001 checkpoint 作为起点：

```text
/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/
handover-block-ppo200-kl001-novideo-4g-m1400-0803-1155/
handover_block_ppo_kl001_fromhb200_128env_4gpu/checkpoints/global_step_300/
actor/model_state_dict/full_weights.pt
```

该 checkpoint 之前两轮含策略采样随机性的 96 条固定种子 hb 评测分别为 62.5% 和 51.0%，均值 56.8%。本轮原计划用 `temperature_eval=-1.0` 改成确定性评测，但代码复查确认 OpenPI 会忽略该参数，初始 action noise 仍随机。

Reference policy：

```text
/mnt/shared-storage-user/chenzanxin/jiayusun/five_task_ppo_step100/full_weights.pt
```

## 三、Stage-2 初始训练：256 env

### 3.1 作业信息

| 项目 | 值 |
|---|---|
| showname | `handover-block-stage2-kl300-256env-8g-n0258-0809-1145` |
| rjob | `handover-block-stage2-kl300-256env-8g-n0258-220bc` |
| 节点 | `gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn` |
| GPU | 8 |
| CPU / memory request | 96 CPU / 1,800,000 MiB |
| 最终状态 | OOM 后停止 |
| 最后完成 step | 27 附近 |
| 最后完整 checkpoint | `global_step_25` |

日志：

```text
/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/
handover-block-stage2-kl300-256env-8g-n0258-0809-1145/run_embodiment.log
```

### 3.2 关键参数

| 参数 | 值 |
|---|---:|
| train env | 256 |
| rollout epoch | 4 |
| 每次 PPO 更新 rollout 数 | 1024 |
| update epoch | 3 |
| global batch size | 4096 |
| micro batch size | 32 |
| actor LR | `2e-6` |
| value LR | `5e-5` |
| `ref_kl_beta` | `0.001` |
| train temperature | `0.8` |
| eval temperature | `-1.0`，但 OpenPI 不使用该参数，eval 仍含初始 action noise |
| eval env / epoch | 32 env × 3 epoch = 96 条 |
| checkpoint / eval interval | 25 step |
| max steps | 150 |

该阶段通过 `runner.ckpt_path=<full_weights.pt>` 加载模型权重，而不是 DCP resume。这样可以保留 checkpoint 中的 value head，同时使用新的 Adam 状态和较低学习率。

### 3.3 首轮同步验证

启用 `weight_syncer.patch.init_sync.enabled=true` 后：

- stage2 step 0 `approx_kl=0.0171`
- 正常训练区间约为 0.008 到 0.022
- `clip_fraction` 约为 5.6% 到 9.1%

这说明 actor checkpoint 已在首轮 rollout 前正确同步，不再出现旧 resume 实验首轮 `approx_kl=10+`、`clip_fraction>0.7` 的异常更新。

### 3.4 OOM 证据

OOM 时日志记录：

```text
Memory on the node: 1685.25 GB / 1757.81 GB (95.87%)
Ray threshold: 95%
```

8 个 EnvWorker 的内存分别约为：

```text
205.52, 196.91, 169.42, 169.21,
163.06, 159.12, 148.93, 147.24 GB
```

EnvWorker 合计约 1359 GB；8 个 actor 合计约 228 GB；rollout workers 约 65 GB。加上 Ray 和系统进程后，单节点没有足够余量。

训练和评测均为 novideo：

- `env.train.video_cfg.save_video=false`
- `env.eval.video_cfg.save_video=false`
- train/eval `eval_video_log=false`
- train/eval `collect_data=false`
- 运行目录内没有 `.mp4`

因此 OOM 的主要来源是 SAPIEN/RoboTwin EnvWorker 中常驻的仿真场景、相机渲染资源和长期缓存，不是视频保存。

## 四、Stage-2 续训：128 env

### 4.1 作业信息

| 项目 | 值 |
|---|---|
| showname | `handover-block-stage2-r25-128env-8g-n0258-0810-1212` |
| rjob | `handover-block-stage2-r25-128env-8g-n0258-08-721ad` |
| 节点 | `gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn` |
| GPU | 8 |
| CPU / memory request | 64 CPU / 1,400,000 MiB |
| resume checkpoint | stage2 `global_step_25` |
| max steps | 150 |
| 当前状态 | Succeeded，已完成 step 150 |

日志：

```text
/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/
handover-block-stage2-r25-128env-8g-n0258-0810-1212/run_embodiment.log
```

checkpoint 根目录：

```text
/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/
handover-block-stage2-r25-128env-8g-n0258-0810-1212/
handover_block_stage2_resume25_128env_8gpu/checkpoints
```

### 4.2 相对 256-env 的修改

| 参数 | 256-env | 128-env |
|---|---:|---:|
| train env | 256 | 128 |
| 每个 EnvWorker 环境数 | 32 | 16 |
| 每次更新 rollout 数 | 1024 | 512 |
| global batch size | 4096 | 2048 |
| rollout epoch | 4 | 4 |
| update epoch | 3 | 3 |
| actor / value LR | `2e-6` / `5e-5` | 不变 |

环境数和 global batch 同比例减半，因此每个 PPO update 的 minibatch/update 结构基本保持一致。主要变化是 rollout 样本方差增大，但 CPU 内存显著下降。

### 4.3 内置评测原始结果

评测使用 handover_block 固定 episode manifest：

```text
rlinf/envs/robotwin/seeds/handover_block_manifest_96ep.json
```

设置：32 eval env × 3 eval rollout epoch，`temperature_eval=-1.0`。该参数对 OpenPI 无效，且同一进程第二次及后续评测存在 manifest 游标耗尽问题。

| Stage2 global step | success_once | `num_trajectories` | 是否有效 96 条 |
|---:|---:|---:|---|
| 25 | 55.2% | 96 | 是，独立进程首次评测 |
| 50 | 55.2% | 96 | 是，resume 进程首次评测 |
| 75 | 59.4% | 544 | 否，约 19/32 被重复统计 |
| 100 | 62.5% | 544 | 否，约 20/32 被重复统计 |
| 125 | 78.1% | 544 | 否，约 25/32 被重复统计 |
| 150 | 59.4% | 544 | 否，约 19/32 被重复统计 |

step 125 到 150 的表面下降为 18.75pp，但有效样本只有最后 32 个环境，且 action noise 随机。该差值不能直接解释为 18/96 个 episode 退化。step 125 仍作为 stage3 候选起点保留，依据还包括 step 100–124 的在线 rollout success 均值高于 step 125–149，而不是只依赖失效的内置评测。

### 4.4 训练指标

128-env 阶段的 rollout `success_once` 大体从 27% 平台提高到 31% 左右：

| step 窗口 | rollout success 均值 |
|---|---:|
| 26–30 | 27.0% |
| 31–35 | 26.6% |
| 51–55 | 29.2% |
| 75–84 | 31.7% |
| 85–94 | 32.0% |
| 100–109 | 31.4% |
| 110–119 | 31.2% |
| 120–126 | 30.7% |

最近训练数值范围：

- `approx_kl`：约 0.010 到 0.023
- `clip_fraction`：约 6% 到 9%
- `value_loss`：多数约 0.04 到 0.06
- `ref_kl_loss`：约 5.4 到 6.1，对 total loss 的贡献约 0.005 到 0.006
- actor LR：`2e-6`
- value LR：`5e-5`

step 125 的随机 rollout 批次成功率较低时，critic `explained_variance` 短暂下降到 -13.1，下一步回升到 -1.39；同期 value loss 仍约 0.039 到 0.057，没有出现 value loss 爆炸。需要继续观察，但当前未造成 PPO KL 或 clip 异常。

## 五、当前判断

1. **128 env 没有导致明显的学习速度下降。** global batch 同比例缩小后，PPO 更新强度保持稳定，同时解决了单节点 OOM。
2. **step 125 的 78.1% 尚未被有效复现。** 它来自最后 32 个 manifest episode 的一次随机评测，不能证明已经达到 75/96。
3. **在线 rollout 有温和退化，而不是训练崩溃。** step 100–124 success 均值为 31.56%，step 125–149 为 28.98%；同期 KL、clip 和 value loss 仍在正常范围。
4. **尚未证明达到 90%。** 需要用新进程覆盖 96 个不同 episode，并检查 `num_trajectories=96`、`episode_len=400`；由于 OpenPI action noise 仍随机，候选 checkpoint 还应做重复评测。

## 六、后续决策

- stage2 已正常结束并保存 step 150；所有旧 checkpoint 均保留。
- 选择 `global_step_125/actor/model_state_dict/full_weights.pt` 启动 stage3，不加载 step 150，也不恢复 DCP optimizer state。
- stage3 使用 fresh optimizer，将 actor LR 从 `2e-6` 降到 `1e-6`，value LR 从 `5e-5` 降到 `2.5e-5`，降低越过高成功率区域后继续退化的风险。
- 其余训练设置保持不变：8 GPU、128 train env、2048 global batch、4 rollout epochs、3 update epochs、reference KL `0.001`、`init_sync=true`、novideo。Stage3 首次周期评测可覆盖 96 条；后续内置评测必须修复或改为独立新进程复评。

### 6.1 Stage3 提交

| 项目 | 值 |
|---|---|
| rjob | `handover-block-stage3-s125-lr1e6-128env-8g-n-0b3f0` |
| showname | `handover-block-stage3-s125-lr1e6-128env-8g-n0258-0811-2324` |
| 节点 | `gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn` |
| 初始权重 | stage2 `global_step_125` full weights |
| optimizer | fresh |
| actor / value LR | `1e-6` / `2.5e-5` |
| train env / GPU | 128 / 8 |
| 计划步数 | 75；step 25 首次评测覆盖 96 条，后续需独立复评 |
| 当前状态 | Running |

## 七、提交脚本

```text
my_docs/submit_handover_block_ppo_stage2_8gpu.sh
my_docs/submit_handover_block_ppo_stage2_resume25_8gpu.sh
my_docs/submit_handover_block_ppo_stage3_from125_8gpu.sh
```

第一个脚本记录 256-env fresh-optimizer 阶段；第二个脚本记录从 stage2 step 25 恢复、改为 128 env 的稳定续训阶段；第三个脚本记录从最佳 step 125 full weights 降低学习率后继续训练的 stage3。
## 八、Stage2 五个 checkpoint 独立复评（2026-08-12）

为排除同一训练进程中 manifest 游标未重置造成的重复统计，重新评测
`global_step_50/75/100/125/150`。五个权重在同一条 rjob 中串行执行，但每个
checkpoint 都会停止并重启 Ray、启动全新的 Python 评测进程。

| 项目 | 值 |
|---|---|
| rjob | `eval-hb-stage2-s50-150-96ep-4g-n0260-0812-01-ee3b2` |
| showname | `eval-hb-stage2-s50-150-96ep-4g-n0260-0812-0100` |
| 指定节点 | `gpu-l-lg-cmc-h-h200-0260.host.h.pjlab.org.cn` |
| GPU / eval env | 4 / 32 |
| checkpoint | Stage2 `global_step_50/75/100/125/150` full weights |
| 每个 checkpoint | 3 rollout epochs，共 96 条固定 manifest episode |
| 提交时状态 | `Inqueue`，节点标签为硬约束 |

节点排队探测：同配置作业曾分别固定提交到 `0256` 和 `0257`，两者也均为
`Inqueue/STARTING`，没有获得 GPU。探测后立即停止这两条作业，仅保留上述
`0260` 作业：

| 节点 | rjob | 最终状态 |
|---|---|---|
| `0256` | `eval-hb-stage2-s50-150-96ep-4g-n0256-0812-01-47760` | `Stopped` |
| `0257` | `eval-hb-stage2-s50-150-96ep-4g-n0257-0812-01-2f617` | `Stopped` |
| `0260` | `eval-hb-stage2-s50-150-96ep-4g-n0260-0812-01-ee3b2` | `Inqueue`，保留 |

结果目录：

```text
/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/
eval-hb-stage2-s50-150-96ep-4g-n0260-0812-0100/
```

汇总文件为 `summary.tsv`，每个 checkpoint 的完整日志在
`step_<N>/run_embodiment.log`。只有同时满足 `num_trajectories=96` 和
`episode_len=400` 的结果才标记为 `valid`；任一结果缺失或不满足条件时，作业
最终返回失败，但会继续尝试后续 checkpoint。提交脚本：

```text
my_docs/submit_eval_handover_stage2_steps_50_150_4gpu.sh
```

## 九、Stage3 首次有效评测（2026-08-12）

Stage3 保存 `global_step_25` 后完成了该进程中的首次评测。TensorBoard 将该次
结果记录在 step 24（训练循环采用零基索引），对应 checkpoint
`global_step_25`：

| checkpoint | success_once | num_trajectories | episode_len | 判定 |
|---|---:|---:|---:|---|
| Stage3 `global_step_25` | 58.33%（56/96） | 96 | 400 | 有效 |

该结果比 Stage2 首次有效评测的 53/96（55.21%）高 3 个成功 episode，但仍远低于
90% 目标；OpenPI 评测仍包含动作噪声，因此单次 3/96 的差异不足以证明稳定提升。
Stage3 PPO 继续运行，所有 checkpoint 保留。Stage2 五权重独立复评作业仍在
`0260` 排队。

### 9.1 Stage3 完成后独立复评

已提交一条固定到 PPO 同一节点 `0258` 的 4-GPU 复评作业。提交时 PPO 尚在
运行，因此该作业保持排队，等待训练释放 GPU；启动后会确认 `global_step_75`
full weights 已完整保存，再用三个独立 Python/Ray 进程串行评测 step 25/50/75。

| 项目 | 值 |
|---|---|
| rjob | `eval-hb-stage3-s25-75-96ep-4g-n0258-0812-200-25363` |
| showname | `eval-hb-stage3-s25-75-96ep-4g-n0258-0812-2005` |
| 节点 | `gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn` |
| checkpoint | Stage3 `global_step_25/50/75` full weights |
| 有效性条件 | 每项 `num_trajectories=96` 且 `episode_len=400` |
| 提交时状态 | `Inqueue` |

结果目录：

```text
/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/
eval-hb-stage3-s25-75-96ep-4g-n0258-0812-2005/
```

提交脚本：`my_docs/submit_eval_handover_stage3_steps_25_75_4gpu.sh`。

### 9.2 Stage3 独立复评结果与 Stage4

Stage3 训练和固定到 `0258` 的独立复评均正常结束。三个 checkpoint 均在独立
进程中覆盖完整 96 条 episode：

| Stage3 checkpoint | success_once | num_trajectories | episode_len |
|---|---:|---:|---:|
| step 25 | 60.42%（58/96） | 96 | 400 |
| step 50 | 57.29%（55/96） | 96 | 400 |
| step 75 | 62.50%（60/96） | 96 | 400 |

最佳为 step 75，但仍未达到 90%。已从 Stage3 step 75 full weights 提交 Stage4：

| 项目 | 值 |
|---|---|
| rjob | `handover-block-stage4-s3s75-lr1e6-128env-8g-5cbd2` |
| showname | `handover-block-stage4-s3s75-lr1e6-128env-8g-n0258-0812-2050` |
| 节点 | `gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn` |
| 初始权重 | Stage3 `global_step_75` full weights |
| optimizer | fresh |
| actor / value LR | `1e-6` / `2.5e-5` |
| train env / GPU | 128 / 8 |
| 计划步数 | 300，每 25 步保存与评测 |
| 提交时状态 | 旧 75-step 作业已停止，由四节点 300-step 队列替代 |

提交脚本：`my_docs/submit_handover_block_ppo_stage4_from_stage3s75_8gpu.sh`。

Stage4 `runner.max_steps` 后续改为 300。为缩短等待时间，同一配置分别固定到四个
批准节点排队；首条进入运行后必须立即停止其余三条，始终只训练一个分支：

| 节点 | rjob | 提交后状态 |
|---|---|---|
| `0256` | `handover-block-stage4-s3s75-300step-8g-n0256-74c87` | `Inqueue` |
| `0257` | `handover-block-stage4-s3s75-300step-8g-n0257-46772` | `Inqueue` |
| `0258` | `handover-block-stage4-s3s75-300step-8g-n0258-135c8` | `Inqueue` |
| `0260` | `handover-block-stage4-s3s75-300step-8g-n0260-4daad` | `Inqueue` |

四条作业均申请 8 GPU、64 CPU、1,400,000 MiB memory。调度事件显示当前没有
目标节点能满足完整资源组合，并明确报告 CPU/内存不足；代理机没有直接登录计算
节点执行 `nvidia-smi` 的权限，因此无法取得其他用户作业的逐卡显存占用。

随后将 memory request 从 1,400,000 MiB 下调到 1,000,000 MiB。四条 1.4 TB
排队作业均在启动前停止，并用以下 1 TB 作业替代：

| 节点 | rjob | 提交后状态 |
|---|---|---|
| `0256` | `handover-block-stage4-s3s75-s300-1tb-8g-n025-4abf4` | `Inqueue` |
| `0257` | `handover-block-stage4-s3s75-s300-1tb-8g-n025-4bfa5` | `Inqueue` |
| `0258` | `handover-block-stage4-s3s75-s300-1tb-8g-n025-8ed5f` | `Inqueue` |
| `0260` | `handover-block-stage4-s3s75-s300-1tb-8g-n026-73f1e` | `Inqueue` |

训练参数不变。任何一条进入运行后立即停止另外三条。

## 十、Stage2 五 checkpoint 独立复评结果

固定到 `0260` 的复评作业已正常结束。每个 checkpoint 均由独立进程评测，且
`num_trajectories=96`、`episode_len=400`：

| Stage2 checkpoint | success_once |
|---|---:|
| step 50 | 51.04%（49/96） |
| step 75 | 58.33%（56/96） |
| step 100 | 53.13%（51/96） |
| step 125 | 54.17%（52/96） |
| step 150 | 62.50%（60/96） |

Stage2 最佳为 step 150，与 Stage3 最佳 step 75 的单次独立复评结果同为
60/96。该结果也确认旧内置评测中 step 125 的 78.1% 是无效高估。Stage4 已从
Stage3 step 75 提交，因为两者评测并列且 Stage3 是当前连续训练链的最新权重；
Stage4 仍在 `0258` 排队。
