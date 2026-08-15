# handover_block 当前试验记录（更新至 2026-07-31 17:17 HKT）

本文记录当前已提交的 `handover_block` PPO 实验。状态会随调度和训练推进变化；最新状态快照见文末。

## 通用运行环境

| 项目 | 设置 |
| --- | --- |
| 镜像 | `registry.h.pjlab.org.cn/ailab-eailabagent-eailabagent_gpu/jiayusun-rlinf-robotwin:openpi-robotwin-vulkan-20260729` |
| 镜像 digest | `sha256:7b47d89ea1e0d93096e8f5b4e2f974018b5d8c11590cde10682937225a298` |
| 资源 | 4 GPU、32 CPU、800,000 MiB RAM |
| 训练环境 | RoboTwin `handover_block`，`ROBOT_PLATFORM=ALOHA` |
| 并行布局 | `actor, env, rollout: 0-3` |
| 训练环境数 | 128（每 GPU 32 个） |
| Eval 环境数 | 32 |
| PPO 参数 | `rollout_epoch=4`、`update_epoch=5`、`global_batch_size=2048`、`micro_batch_size=32` |
| 清理策略 | `env.train.task_config.clear_cache_freq=2` |
| 资产 | `/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf` |
| 运行产物 | `gpfs2://ailab-eailabagent-gpfs` 挂载至 `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs` |
| rjob 配额 | `eailabagent_gpu` |

所有任务都使用镜像内 Python 环境：

```bash
source /opt/venvs/openpi-robotwin-cu124/bin/activate
```

## 实验 A：step-200 无参考 KL 续训

| 项目 | 设置 |
| --- | --- |
| rjob | `handover-block-ppo200resume-4g-m800-0729-205-e08c7` |
| 最新状态 | `Stopped`；此前 Ray OOM 后停在 `global_step=270` 附近，最后可用 checkpoint 为 `global_step_250` |
| 起点 | handover_block PPO `global_step_200` |
| checkpoint | `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-checkpoints/handover_block_global_step_200` |
| 最大 step | 500 |
| KL | 无参考策略正则（`ref_kl_beta=0`） |
| 日志/新 checkpoint | `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo200resume-4g-m800-0729-2052` |
| 提交脚本 | [`submit_handover_block_ppo200resume_openpi_vulkan_4gpu.sh`](submit_handover_block_ppo200resume_openpi_vulkan_4gpu.sh) |

## 实验 B：step-200 + 5-task 参考 KL（0.01）

| 项目 | 设置 |
| --- | --- |
| rjob | `handover-block-ppo200-kl01-fix2-4g-m800-0730-1d2d9` |
| 最新状态 | `Succeeded`，但日志显示 Ray OOM kill，应视为异常/假成功 |
| 起点 | 与实验 A 相同的 `global_step_200` DCP checkpoint |
| 最大 step | 500 |
| 参考策略 | 5-task joint PPO step-100 |
| 参考权重 | `/mnt/shared-storage-user/chenzanxin/jiayusun/five_task_ppo_step100/full_weights.pt` |
| KL 参数 | `kl_penalty=low_var_kl`、`ref_kl_beta=0.01` |
| 兼容性 | 参考权重与 step-200 actor checkpoint 均为 821 个 key，已验证 key 集完全一致 |
| 关注指标 | TensorBoard `actor/ref_kl_loss` |
| 日志/新 checkpoint | `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo200-kl01-fix2-4g-m800-0730-1218` |
| 配置 | [`moe_sft20k_handover_block_ppo_kl01_fromhb200_4gpu.yaml`](../examples/embodiment/config/moe_sft20k_handover_block_ppo_kl01_fromhb200_4gpu.yaml) |
| 提交脚本 | [`submit_handover_block_ppo200resume_kl01_openpi_vulkan_4gpu.sh`](submit_handover_block_ppo200resume_kl01_openpi_vulkan_4gpu.sh) |

备注：早期 KL job `handover-block-ppo200-kl01-4g-m800-0729-2137-e4830` 和第一次修复 job 不是有效训练。有效训练是 `fix2`：提交脚本使用 Hydra 追加字段 `+algorithm.ref_kl_beta=0.01`、`+algorithm.ref_policy_path=...`。

## 实验 C：step-50 无参考 KL 续训

| 项目 | 设置 |
| --- | --- |
| rjob | `handover-block-ppo50resume-4g-m800-0729-2138-0d859` |
| 最新状态 | `Succeeded`，但日志显示 Ray OOM kill，应视为异常/假成功 |
| 起点 | handover_block PPO `global_step_50` |
| checkpoint | `/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin/logs/20260725-11:32:49-moe_sft20k_handover_block_ppo150_128env_4gpu/moe_sft20k_handover_block_ppo150_128env_4gpu/checkpoints/global_step_50` |
| checkpoint 格式 | PyTorch Distributed Checkpoint：4 个 DCP 分片、`.metadata` 和 `full_weights.pt` 已验证存在 |
| 最大 step | 500 |
| KL | 无参考策略正则（`ref_kl_beta=0`） |
| 日志/新 checkpoint | `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo50resume-4g-m800-0729-2138` |
| 提交脚本 | [`submit_handover_block_ppo50resume_openpi_vulkan_4gpu.sh`](submit_handover_block_ppo50resume_openpi_vulkan_4gpu.sh) |

## 2026-07-30 20:15 HKT 监控快照

| 实验 | rjob 状态 | 最新证据 | 判断 |
| --- | --- | --- | --- |
| A：step-200 无参考 KL 续训 | `Running` | `run_embodiment.log` 更新到 `2026-07-30 20:14:38`；train video 到 `seed_*/270.mp4`；checkpoint 已有 `global_step_250` | 正常训练中 |
| B：step-200 + 5-task 参考 KL（0.01） | `Running` | `run_embodiment.log` 更新到 `2026-07-30 20:15:02`；train video 到 `seed_*/66.mp4`；rollout 日志显示 `Generating Rollout Epochs: 75% 3/4` | 正常训练中 |
| C：step-50 无参考 KL 续训 | `Succeeded` | `run_embodiment.log` 最后更新到 `2026-07-30 18:42:25`；checkpoint 只有 `global_step_100`；train video 到 `seed_*/235.mp4` | 非完整完成；Ray OOM 后假成功 |

实验 C 的关键异常：

```text
2 worker(s) were killed due to the node running low on memory
Memory on the node was 756.42GB / 781.25GB
OOM kill reason: Memory usage 797959262208B exceeded threshold of 796917760000B
Top EnvWorker memory: 182.44GB, 172.30GB, 163.76GB, 113.54GB
```

结论：

- 实验 A 和 B 当前继续运行，不建议停止。
- 实验 C 没有跑到 `runner.max_steps=500`，只产生到 `global_step_100`，后续若要继续需要从 `global_step_100` 断点重提，并解决 128 env 下 EnvWorker CPU RAM 增长问题。
- 3 个任务日志中都出现过 `curobo.types.math` 导入异常，但随后环境构建成功并进入 rollout，因此当前不是阻塞原因。

## 实验 D：step-100 无视频续训

| 项目 | 设置 |
| --- | --- |
| rjob | `handover-block-ppo100resume-novideo-4g-m800-abbf1` |
| showname | `handover-block-ppo100resume-novideo-4g-m800-0730-2024` |
| 提交时间 | `2026-07-30 20:36 HKT` |
| 最新状态 | `Stopped`，用户取消 |
| 起点 | 实验 C 产出的 `global_step_100` |
| checkpoint | `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo50resume-4g-m800-0729-2138/handover_block_ppo50resume_128env_4gpu/checkpoints/global_step_100` |
| 最大 step | 500 |
| 资源 | 4 GPU、32 CPU、800,000 MiB RAM |
| 环境数 | train 128、eval 32 |
| 视频 | 已关闭：`env.train.video_cfg.save_video=False`、`env.eval.video_cfg.save_video=False`、`env.train.task_config.eval_video_log=false`、`env.eval.task_config.eval_video_log=false` |
| 日志/新 checkpoint | `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo100resume-novideo-4g-m800-0730-2024` |
| 提交脚本 | [`submit_handover_block_ppo100resume_novideo_openpi_vulkan_4gpu.sh`](submit_handover_block_ppo100resume_novideo_openpi_vulkan_4gpu.sh) |

目的：验证关闭 `RecordVideo` 和 RoboTwin eval video logging 后，是否能降低长时间 128-env handover_block rollout 的内存增长，避免实验 C 在 `global_step_100` 后出现 Ray OOM kill。

## 实验 E：step-250 无参考 KL、无 image/video 续训

| 项目 | 设置 |
| --- | --- |
| rjob | `handover-block-ppo250resume-novideo-4g-m800-e61fc` |
| showname | `handover-block-ppo250resume-novideo-4g-m800-0731-1539` |
| 提交时间 | `2026-07-31 15:40 HKT` |
| 最新状态 | `Running`，节点 `gpu-lg-cmc-h-h200-1798` |
| 起点 | 实验 A 产出的 `global_step_250` |
| checkpoint | `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo200resume-4g-m800-0729-2052/handover_block_ppo200resume_128env_4gpu/checkpoints/global_step_250` |
| 最大 step | 500 |
| KL | 无参考策略正则：`algorithm.kl_penalty=kl`、`algorithm.kl_beta=0.0` |
| 资源 | 4 GPU、32 CPU、800,000 MiB RAM |
| 环境数 | train 128、eval 32 |
| image/video | 已关闭：`env.train.video_cfg.save_video=False`、`env.eval.video_cfg.save_video=False`、`env.train.task_config.eval_video_log=false`、`env.eval.task_config.eval_video_log=false`、`collect_data=false`、`save_freq=null` |
| 日志/新 checkpoint | `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo250resume-novideo-4g-m800-0731-1539` |
| 提交脚本 | [`submit_handover_block_ppo250resume_novideo_openpi_vulkan_4gpu.sh`](submit_handover_block_ppo250resume_novideo_openpi_vulkan_4gpu.sh) |

目的：从无 KL 的 `global_step_250` 继续训练到 500，同时关闭 RLinf `RecordVideo` 和 RoboTwin 侧 image/video/data 保存，验证是否能减缓 128-env 长时间 rollout 的 CPU RAM 增长。

## 2026-07-31 17:17 HKT 监控快照

| 实验 | rjob 状态 | 最新证据 | 判断 |
| --- | --- | --- | --- |
| A：step-200 无参考 KL 续训 | `Stopped` | `global_step_250` checkpoint 存在；此前最后训练指标约 `Global Step: 270/500`；日志出现 Ray OOM，EnvWorker 最高约 `170GB` | 已停止；可用断点为 `global_step_250` |
| B：step-200 + 5-task 参考 KL（0.01） | `Succeeded` | `run_embodiment.log` 最后更新到 `2026-07-31 13:24:31`；checkpoint 只有 `global_step_250`；日志有 `3 Workers killed due to memory pressure (OOM)` | 非完整完成；Ray OOM 后假成功 |
| C：step-50 无参考 KL 续训 | `Succeeded` | checkpoint 只有 `global_step_100`；日志有 Ray OOM | 非完整完成；Ray OOM 后假成功 |
| D：step-100 无视频续训 | `Stopped` | 用户取消 | 不再监控 |
| E：step-250 无参考 KL、无 image/video 续训 | `Running` | `run_embodiment.log` 更新到 `2026-07-31 17:15:03`；TensorBoard 更新到 `2026-07-31 17:05:30`；rollout 日志显示 `Generating Rollout Epochs: 50% 2/4`；未看到 video 文件产出 | 当前唯一正常运行中的无 KL 续训 |

实验 B 的关键异常：

```text
3 Workers (tasks / actors) killed due to memory pressure (OOM)
Exception occurred while running EmbodiedFSDPActor's function recv_rollout_trajectories
```

实验 E 当前结论：

- 已正确从实验 A 的 `global_step_250` 断点恢复。
- 当前未看到 `video/train/*.mp4` 产物，说明 no-video/no-image 设置按预期生效。
- 仍使用 128 train env 和 800,000 MiB RAM；是否能真正避免 OOM 需要继续观察到多个 step/checkpoint。

## 实验 F：step-200 + 5-task 参考 KL（0.001，诊断实验）

| 项目 | 设置 |
| --- | --- |
| rjob（第一次，m800） | `handover-block-ppo200-kl001-novideo-4g-m800-45365` | `Stopped`（系统升级杀掉，跑到 step 211） |
| rjob（重提交，m1400） | `handover-block-ppo200-kl001-novideo-4g-m1400-2f3f1` | 人工在 2026-08-05 12:41 HKT 停止，最终停在 step 304/500，保存了 `global_step_250`、`global_step_300` |
| showname | `handover-block-ppo200-kl001-novideo-4g-m1400-0803-1155` |
| 起点 | 与实验 A/B 相同的 `global_step_200` DCP checkpoint |
| 参考策略 | 5-task joint PPO step-100（同实验 B） |
| KL 参数 | `kl_penalty=low_var_kl`、`ref_kl_beta=0.001`（实验 B 的 1/10） |
| 最大 step | 500 |
| image/video | 已关闭（沿用实验 E 设置） |
| 日志/新 checkpoint | `/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo200-kl001-novideo-4g-m1400-0803-1155`（注意：系统升级后开发机本机 gpfs2 挂载路径从 `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs` 变为 `/mnt/ailab-eailabagent-gpfs`，容器内部路径不受影响） |
| 提交脚本 | [`submit_handover_block_ppo200resume_kl001_openpi_vulkan_4gpu.sh`](submit_handover_block_ppo200resume_kl001_openpi_vulkan_4gpu.sh) |

目的：实验 B（`ref_kl_beta=0.01`）数据分析显示 KL 惩罚项量级 ≈ PPO 信号量级的一半（0.01×4.46≈0.045 vs 纯 PPO policy_loss ≈0.092），压制了 success_once 提升趋势，但幅度不足以让 `ref_kl_loss` 明显下降。本实验把 β 降到 0.001（KL 项约为 PPO 信号的 5%），验证问题是否纯粹是力度过大。

**训练时 on-policy 指标（2026-08-04 更新，51-step 窗口 202–251）**：success_once 均值 0.255，介于 A（0.282，明显上升）和 B（0.241，明显下降）之间，趋势基本走平（不再下降，但也没恢复上升）；`ref_kl_loss` 基本不再收缩。

**5-task 固定种子评测对比（2026-08-05，checkpoint `global_step_300`，vs 无 KL 实验 E 同 step）**：第一轮显示 KL 组在 handover_block 上明显更高（+7.3pp）、click_alarmclock 上明显更差（-9.1pp）。但重复评测同一 checkpoint 第二轮后（eval 配置 `do_sample=True` 导致策略采样有随机性），发现单次评测的波动幅度（普遍 5~9pp，个别到 11~12pp）跟第一轮观察到的差异是同一量级——两轮取均值后，五任务总均值基本打平（KL 72.8% vs 无KL 72.6%，+0.2pp），hb 上 KL 只领先 3.1pp，ca 上反而落后 6.7pp。**结论：目前能确认的只有"β=0.001 没有拖累整体多任务能力"，"KL 帮助了 handover_block"这个更强的说法尚未被证实，需要更多轮次评测。** 详细分析见 [`2026-07-31-kl-impact-analysis-and-kl001-decision.md`](2026-07-31-kl-impact-analysis-and-kl001-decision.md)。

同期无 KL 对照实验 E 也因系统升级被杀（`Failed`），重提交为 `handover-block-ppo250resume-novideo-4g-m1400-058d8`（`Running`，截至 2026-08-05 14:49 HKT，step 400/500），内存同样从 800000 提到 1400000。

## 已停止的对照作业

`handover-block-ppo-fromsft-4g-m800-0729-1913-930cc` 已于 21:37 HKT 停止。该作业原本从 SFT-20k 权重 step 0 开始训练，停止后由实验 C 的 step-50 续训替代。
