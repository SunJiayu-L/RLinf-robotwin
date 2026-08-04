# PPO50→500 续训启动阶段卡死 / raylet 被杀 排查记录

日期：2026-07-27 ~ 2026-07-28

## 1. 背景

- 原始 4 卡 PPO 任务（`moe_sft20k_handover_block_ppo150_128env_4gpu`，无 `resume_dir`，从 `sft20k` 全新开始）在 **H200-1798** 上跑到 step 88，因 CPU RAM 触达 Ray 的 95% 阈值被 OOM kill 停止。已保存可用的 `global_step_50` checkpoint：
  ```
  /mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin/logs/20260725-11:32:49-moe_sft20k_handover_block_ppo150_128env_4gpu/moe_sft20k_handover_block_ppo150_128env_4gpu/checkpoints/global_step_50
  ```
- 目标：从 `global_step_50` 续训到 step 500（`runner.resume_dir=<step50>` + `runner.max_steps=500`）。
- 结果：**此后所有续训尝试（以及后来连原始"全新训练"配置原样重跑）在启动阶段全部失败**，症状高度一致：4 个 rollout rank 打印完 `"Rollout worker initialized with dst_ranks/src_ranks"` 后，日志戛然而止，通常伴随
  ```
  (raylet) Raylet is terminated. Termination is unexpected. Possible reasons include:
  (1) SIGKILL by the user or system OOM killer, (2) SIGSEGV/SIGBUS, (3) Other termination signals.
  ```
  之后 `rjob` 状态可能长期显示 `RUNNING`（driver 进程变成孤儿，Ray 集群已死但容器还没退出），也可能过一段时间自动变成 `Stopped`（容器 `trap ... ray stop --force; exit` 触发）。

## 2. 已尝试的组合与结果

| 时间 | 任务名 | GPU | 内存 | rollout 权重来源 | 关键参数 | 物理节点 | 结果 |
|---|---|---|---|---|---|---|---|
| 07-27 11:35 | hb-ppo50to500-4g-m1400-0727-1135 | 4 | 1,400,000 | step50 `full_weights.pt` 直接加载 | `resume_dir` | h200-0685 | 卡死（Rollout init 后, raylet 被 SIGKILL） |
| 07-27 13:20 | hb-ppo50to500-4g-m1400-sync-0727-1320 (及 clone 1349/1357) | 4 | 1,400,000 | sft20k | `resume_dir` + `init_sync=true` | h200-0685（重试 3 次全落同一物理机） | 卡死，同样症状 |
| 07-27 14:40 | hb-ppo50to500-4g-m800-0727-1440 | 4 | 800,000 | sft20k | `resume_dir` + `init_sync=true` | h200-0685 | 卡死，同样症状 |
| 07-27 15:24 | hb-ppo50to500-4g-m800-0727-1524 | 4 | 800,000 | sft20k | `resume_dir` + `init_sync=true` | h200-0685（第5次同节点） | 卡死，同样症状 |
| 07-27 16:11 / 17:00 | hb-ppo50to500-8g-m1400 / m800 | 8 | 1,400,000 / 800,000 | sft20k | `resume_dir` + `init_sync=true` | m1400 版本资源不足排不上；m800 版本落在 **h200-1870**（首次不同节点） | m800 版本同样卡死，症状完全一致 |
| 07-27 22:28 | hb-ppo50to500-4g-m800-fix-0727-2033 | 4 | 800,000 | **`runner.ckpt_path=<step50>/full_weights.pt`**（去掉 `init_sync`） | `resume_dir` + `ckpt_path` | h200-1798 | 卡死，同样症状 |
| 07-28 00:20 | hb-ppo50to500-4g-m800-fix2-0728-0021 | 4 | 800,000 | 无（保留 sft20k 默认）| **仅 `resume_dir`**，去掉 `ckpt_path` 和 `init_sync` | h200-1798（同节点再跑） | 卡死，同样症状，日志中从未出现 `"Resuming training from checkpoint directory..."`（即崩溃发生在 DCP 恢复代码真正被调用**之前**） |
| 07-28 02:18 | **hb-control-fresh-0728-0128**（对照组：**完全照搬 3 天前成功的原始配置，无任何 resume/ckpt_path/init_sync**） | 4 | 800,000 | sft20k（原生流程） | 无 | h200-0685 | **同样卡死，同样症状** |
| 07-28 11:09 | hb-ppo50to500-4g-env64-0728-0300 | 4 | 800,000 | sft20k | `resume_dir` + **`env.train.total_num_envs=64`**（从 128 调低一半，验证是否是启动阶段并发构建环境的内存峰值问题） | h200-1798（第2次同节点） | **同样卡死，同样症状**（甚至这次 4 个 rollout rank 里只有 3 个来得及打印 init 日志，raylet 就被杀了，比之前更快） |

## 3. 代码层面排查（已证伪的假设）

用 Explore 子代理深入读了 `rlinf/runners/embodied_runner.py`、`rlinf/hybrid_engines/fsdp/strategy/base.py`、`rlinf/hybrid_engines/weight_syncer/patch_syncer.py`、`rlinf/scheduler/collective/collective_group.py`、`rlinf/workers/rollout/hf/huggingface_worker.py`、`rlinf/workers/actor/fsdp_actor_worker.py`：

1. **假设：`weight_syncer.patch.init_sync.enabled=true` 导致 actor/rollout 之间的裸 NCCL 握手死锁。**
   - 支持证据：该配置文件里 `weight_syncer.patch.init_sync.enabled` 默认就是 `False`，且代码注释明确写着 *"Actor and rollout load the same checkpoint, so the initial full-model handshake is redundant **and can deadlock before the first rollout**."*（`examples/embodiment/config/moe_sft20k_handover_block_ppo150_128env_4gpu.yaml:22-27`）。之前的续训脚本用 CLI 覆盖把它打开成了 `true`。
   - **证伪**：去掉 `init_sync=true`（改用 `runner.ckpt_path` 直接加载权重，见下条）之后依然在同一位置卡死；再去掉 `ckpt_path`、只保留 `resume_dir`，依然卡死；日志里始终没出现 `weight_syncer` 相关的任何 sync 日志，说明崩溃点在 `update_rollout_weights()`（`embodied_runner.py:280-282`，这是第一次触发 actor↔rollout 跨组 collective 的地方）**之前**，`init_sync` 根本没被跑到。

2. **假设：`runner.ckpt_path` 导致 actor（`fsdp_actor_worker.py:1039-1040`）和 rollout（`huggingface_worker.py:100-101`）各自额外 `torch.load()` 一份完整权重，8 个 rank 并发峰值内存过大。**
   - 这是真实存在的多余开销（actor 明明会走 DCP 恢复，`ckpt_path` 对它是纯浪费），已经从脚本里去掉。
   - **证伪为唯一根因**：去掉 `ckpt_path` 后依然卡死。

3. **假设：DCP checkpoint 恢复（`torch.distributed.checkpoint.load()`，`strategy/base.py:270-358`）本身有问题（比如缺少 `barrier()` 导致某个 rank 抛异常时其他 rank 卡在 collective 里）。**
   - **证伪**：日志里从未打印 `embodied_runner.py:152` 的 `"Resuming training from checkpoint directory..."`（这行日志在调用 `load_checkpoint()` 之前打），说明程序**根本没走到**这段代码就已经死了。`resume_dir` 分支自身没有机会成为死锁点。

4. **决定性对照实验**：完全去掉一切续训相关参数（`resume_dir`/`ckpt_path`/`init_sync` 全部不设置），**原样复刻 3 天前跑到 step 88 的那个配置**重新提交，**依然在完全相同的位置死掉**（`hb-control-fresh-0728-0128`，2026-07-28 02:18，节点 h200-0685）。

   → **这彻底排除了"续训配置本身有 bug"这个方向**。问题不在 `resume_dir`/`init_sync`/`ckpt_path`，而在**当前环境本身**。

## 4. 当前结论

- 两天内在 **5 台不同物理机**（h200-0685 ×3、h200-1798 ×3、h200-1870 ×1）上，**9 次尝试（含 1 次对照组 + 1 次 `total_num_envs=64` 调低测试）全部在同一位置失败**：4 个 rollout rank 陆续完成模型初始化的那个时间点附近，Ray 的 raylet 进程被 SIGKILL（大概率是宿主机级别 OOM killer，不是我们容器自己 `--memory` 限制触发的，因为容器内 cgroup OOM 会有不同的、更"干净"的容器内报错，而不是 raylet 直接被外部信号打死）。
- 因为**换配置、换内存申请量（800G/1.4T）、换 GPU 数（4/8）、换物理机、换回原始"从 sft20k 全新训练"配置、甚至把并发环境数从 128 砍到 64，全部无法避免**，唯一站得住的解释是：**这几天私有机器组（`--charged-group=eailabagent_gpu --private-machine=group`）里的物理机实际可用内存/资源，比 3 天前原始成功跑的那次要紧张得多**（可能是同组内其他任务在共享这些物理机、持续抢占内存，也可能是这批节点近期普遍资源紧张/退化，甚至不排除节点硬件/内核层面的问题）。这是**环境/资源层面**的问题，**不是** RLinf-robotwin 训练代码、续训逻辑、或本次训练配置本身的 bug——我们已经把所有能在应用层控制的变量都排除了。

## 5. 已排除的因素（不用再纠结）

- `curobo.types.math` 的 `ModuleNotFoundError` 导入报错——每次都出现，会自动回退到 MPlib planner，非致命，原始成功的跑法（0→88 step）也一样有这个报错，跟卡死无关。
- rollout 权重来源（sft20k / 直接读 step50 `full_weights.pt` / `runner.ckpt_path`）——三种都试过，都卡死。
- `weight_syncer.patch.init_sync.enabled`（true/去掉恢复默认 false）——都卡死。
- 内存申请量 800000 / 1,400,000 MiB——都卡死（且 8卡+1.4T 的组合在当前集群资源下甚至直接排不上号，`rjob events` 显示 `Insufficient nvidia.com/gpu` / `Insufficient memory`）。
- GPU 数量 4 / 8——都卡死。
- 物理机——5 台不同节点都复现。
- `env.train.total_num_envs`（128 → 64）——同样卡死，且这次连 rollout 初始化都没能全部打完 4 个 rank 的日志，raylet 就被杀了，说明启动阶段的内存峰值 / 资源压力问题跟并发环境数量关系不大（或者说，就算把这个变量降到一半，环境本身的资源紧张程度也已经压不住）。

## 6. 下一步计划

**应用层能控制的变量已经全部测完，且全部失败。** 结合"完全复刻 3 天前成功的配置也照样失败"这条决定性证据，问题不在我们这边能改的任何训练参数上，继续在配置上试探已经没有意义。

下一步建议**直接联系飞书技术支持群 / 集群管理员**，反馈私有机器组（`eailabagent_gpu`）近期节点资源疑似紧张/异常，并附上：
- 本文档第 2 节的完整复现记录（9 次尝试，5 台不同物理机，全部同一症状）
- 任意一次失败任务的访问链接 / JOB_ID（例如 `hb-ppo50to500-4g-env64-0728-0300-53976431`，节点 `gpu-lg-cmc-h-h200-1798.host.h.pjlab.org.cn`）
- 完整启动命令（见 `my_docs/submit_handover_block_ppo50to500_4gpu_resume_m1400.sh`）
- 失败日志关键片段："Rollout worker initialized" 后立刻 "Raylet is terminated...SIGKILL by the user or system OOM killer"
- 强调：**3 天前用同一套镜像/脚本/配置在 h200-1798 上能跑到 step 88，现在原样复现即失败**，怀疑节点侧近期资源变化，而非我们训练代码/配置的问题。

如果管理员确认是节点资源被其他任务占用，可以等资源释放后重试，或申请专属节点/更高优先级排除干扰后再跑。

## 7. 涉及的脚本

- `my_docs/submit_handover_block_ppo50to500_4gpu_resume_m1400.sh`：4卡续训脚本，已改为仅保留 `runner.resume_dir`，去掉了 `weight_syncer.patch.init_sync.enabled=true` 和 `runner.ckpt_path`；当前额外带着 `env.train.total_num_envs=64`（诊断测试遗留，若要恢复正常续训需要把这行删掉或改回 128）。内存当前为 800000。
- `my_docs/submit_handover_block_ppo50to500_8gpu_resume_m1400.sh`：8卡续训脚本（新建于本次排查），内存当前为 800000。
- `my_docs/submit_handover_block_ppo_4gpu.sh`：原始"从 sft20k 全新训练"脚本，用作本次的对照组，未做任何修改。
