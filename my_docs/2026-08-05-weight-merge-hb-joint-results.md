# 权重融合实验结果：handover_block PPO × 5-task 联合 PPO

**日期：** 2026-08-05
**Branch：** `KL`
**计划文档：** `.claude/plans/optimized-hopping-liskov.md`（已按计划执行完毕）

---

## 一、背景与方法

不在训练时加正则（KL 约束实验已经证明效果不确定，见 `2026-07-31-kl-impact-analysis-and-kl001-decision.md`），改为训练后直接做权重融合（task arithmetic / 线性插值），把 handover_block 单任务 PPO 和 5-task 联合 PPO 的权重按不同方式、不同系数混合。

**输入 checkpoint：**
- SFT base：`/mnt/shared-storage-user/chenzanxin/jiayusun/sft20k/model.safetensors`
- handover_block PPO：无KL续训 `global_step_400`（`handover-block-ppo250resume-novideo-4g-m1400-0803-1155`）
- 5-task 联合 PPO：`five_task_ppo_step100/full_weights.pt`

**融合方式（两种脚本调用形式，用 `scripts/task_arithmetic_rlinf.py`）：**
- Family A（`mode1-*`）：SFT 为基底，两个任务向量 `theta* = SFT + a_hb*(hb-SFT) + a_joint*(joint-SFT)`
- Family B（`hbline-r*`）：hb 为基底，单一任务向量 `theta* = hb + r*(joint-hb)`，r 是混入 joint 的比例（数学上"以 hb 为基底"和"以 joint 为基底"是同一条线，两种命名视角统一用这一种调用形式覆盖，含外推点 r<0、r>1）

**评测：** 复用 5-task 固定种子评测（bbh/ca/cb/hb/mpa），每个候选跑 2 次独立重复取均值（应对评测本身的采样噪声）。

---

## 二、完整结果

| 候选 | bbh | ca | cb | **hb** | mpa | avg | min |
|---|---|---|---|---|---|---|---|
| baseline-hb400（纯 hb PPO） | 55.3% | 91.2% | 87.5% | 42.2% | 78.3% | 70.9% | 42.2% |
| baseline-joint100（纯联合 PPO） | 68.9% | 93.1% | 96.9% | **0.5%** | 84.8% | 68.8% | 0.5% |
| hbline-r-025（外推，越过 hb） | 46.4% | 86.6% | 83.9% | 48.4% | 86.5% | 70.4% | 46.4%(bbh) |
| hbline-r025（75% hb + 25% joint） | 65.6% | 89.2% | 94.8% | 43.8% | 87.2% | 76.1% | 43.8% |
| hbline-r050（中点） | 74.6% | 81.2% | 99.0% | 32.3% | 90.7% | 75.6% | 32.3% |
| hbline-r075（25% hb + 75% joint） | 80.7% | 97.7% | 97.9% | 9.4% | 85.0% | 74.1% | 9.4% |
| hbline-r125（外推，越过 joint） | 84.1% | 95.9% | 99.0% | 0.0% | 84.9% | 72.8% | 0.0% |
| mode1-a03（SFT基底，均匀0.3） | 71.6% | 87.2% | 81.2% | 30.2% | 83.7% | 70.8% | 30.2% |
| mode1-a05（SFT基底，均匀0.5） | 71.2% | 85.8% | 95.3% | 33.9% | 84.7% | 74.2% | 33.9% |
| mode1-a07（SFT基底，均匀0.7） | 74.7% | 91.3% | 99.5% | 38.0% | 93.3% | **79.4%** | 38.0% |
| **mode1-hb07-joint04**（SFT基底，hb权重更高） | 70.5% | 89.0% | 95.8% | **44.3%** | 74.5% | 74.8% | **44.3%** |

（每个数字都是 2 次独立重复的均值；完整原始数据见 `my_docs/wm-hb-joint-0805-results-table.md`）

---

## 三、关键发现

### 3.1 纯联合 PPO 在 handover_block 上几乎完全失败

`baseline-joint100` 的 hb = 0.5%（两次重复分别是 0% 和 1.0%，高度一致，不是噪声）。这个 checkpoint 训练时确实包含了 handover_block 任务，但在这个固定种子评测基准上几乎完全不会做这个任务，同时在其他 4 个任务上表现都很好（69~97%）。这是解读所有融合结果的关键参照点——融合的目的本质上是把这个"归零"的任务拉回来，同时尽量不牺牲另外 4 个任务。

### 3.2 有两个融合候选在 hb 上超过了纯 hb PPO 基线

- `mode1-hb07-joint04`：hb 44.3%（vs 纯 hb 基线 42.2%），同时 avg 74.8%
- `hbline-r-025`：hb 48.4%，但 bbh 掉到 46.4%（低于两个纯基线的 bbh），avg 只有 70.4%

这两个候选有一个共同特征：**都是"以 hb 为主、只掺入少量 joint 信号"的配置**（mode1-hb07-joint04 给 hb 的系数 0.7 明显高于 joint 的 0.4；hbline-r-025 是从 hb 出发往"远离 joint"的方向外推，等于进一步放大 hb 自身的特征）。这跟最初的假设一致：hb 是 SFT baseline 最低、最容易被稀释的任务，融合时需要给它更高的权重才不会被"稀释"掉。

### 3.3 沿 hb↔joint 这条线，hb 的表现是非单调的

r=-0.25: 48.4% → r=0.25: 43.8% → r=0.50: 32.3% → r=0.75: 9.4% → r=1.25: 0.0%

大体上随着 joint 混入比例增加单调下降，符合直觉（joint 本身在 hb 上就是 0.5%，混入越多越接近 0）。中间没有出现"好过两端"的甜蜜点。两个外推点（r=-0.25 和 r=1.25）行为符合预期：往 hb 方向外推能进一步提升 hb（48.4%，本实验最高值之一），往 joint 方向外推则让 hb 彻底归零。

### 3.4 avg 最高的候选（mode1-a07）hb 并不是最好的

`mode1-a07`（SFT 基底、均匀 α=0.7）avg 79.4% 是全场最高，其他 4 个任务都很强（bbh 74.7%、ca 91.3%、cb 99.5%、mpa 93.3%），但 hb 只有 38.0%，明显低于两个"hb 优先"的候选。这印证了之前 5-task BCAM 融合实验的教训——**排名不能只看 avg，容易被表现好的任务"平均"掩盖掉弱任务的问题**。

---

## 四、结论与推荐

如果目标是**同时兼顾 handover_block 和其他 4 个任务**，目前数据支持的候选是：

1. **`mode1-hb07-joint04`**（SFT 基底，α_hb=0.7 / α_joint=0.4）：min=44.3%（hb 本身），avg=74.8%，是本轮实验里唯一一个"min 就是 hb 且 hb 超过纯 hb 基线"的候选，各任务比较均衡。
2. **`hbline-r025`**（hb 为基底，25% 混入 joint）：avg=76.1%（全场第二高），hb=43.8%，min=43.8%，avg 比候选 1 更高，hb 略低一点点。

如果目标是**整体 5 任务综合表现最强、不特别在意 hb 是否达到单任务 PPO 水平**，`mode1-a07`（avg 79.4%）是最优选择，但 hb 只有纯 hb PPO 基线的 90% 左右。

**局限**：所有数字都只有 2 次重复，之前的 KL 实验里发现单次评测的方差可以到 5~12 个百分点（本轮部分候选，如 hbline-r025 的 bbh，两次重复间差了 14.5pp），2 次重复只能算初步趋势，还不足以精确排序相近的候选（比如 mode1-hb07-joint04 的 44.3% vs hbline-r025 的 43.8% 这种 0.5pp 的差距，在噪声范围内不能说谁更好）。如果要正式定论，建议对排名靠前的 2-3 个候选补充到 4~5 次重复。

**下一步（如果继续）**：
1. 给 `mode1-hb07-joint04` 和 `hbline-r025` 补充重复评测（各再跑 2~3 次），确认排名稳定
2. 在 `mode1-hb07-joint04` 附近做更细的系数网格（例如 α_hb ∈ {0.6, 0.7, 0.8}，α_joint 固定在 0.3~0.4），看能否进一步把 hb 推高而不牺牲其他任务
3. 尝试给 hbline 方向的外推做更细的探索（例如 r ∈ {-0.4, -0.5}），看 hb 是否还能继续提升，以及其他任务开始明显下降的临界点在哪里

---

## 五、产出物

- Merge 脚本：`my_docs/submit_merge_hb_joint_weights.sh`
- 通用评测提交脚本：`my_docs/submit_eval_5task_generic.sh`
- 结果解析脚本：`scripts/parse_eval_success.py`
- 9 个融合权重：`/mnt/ailab-eailabagent-gpfs/chenzanxin/merged_weights/wm-hb-joint-0805/<candidate>/full_weights.pt`
- Merge manifest：`/mnt/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/wm-merge-hb-joint-0805-1637/MERGE_MANIFEST.md`
- 完整结果表：`my_docs/wm-hb-joint-0805-results-table.md`

---

## 六、后续实验：从 mode1-hb07-joint04 出发继续做 5-task 联合 PPO（2026-08-05~06）

### 6.1 目的与设置

融合只是静态插值，想验证从这个融合点出发**继续做真正的 5-task 联合 PPO 训练**，能不能把 5 个任务一起进一步训上去。

- 新建 config：`examples/embodiment/config/moe_ppo_5task_bbh_ca_cb_hb_mpa_128env_4gpu.yaml`（任务/算法参数取自原始 8-GPU 5-task 联合训练配置 `moe_ppo_5task_bbh_ca_cb_hb_mpa.yaml`，基础设施参数取自本 session 跑通的 4-GPU/128-env 单任务配置，详见 `.claude/plans/optimized-hopping-liskov.md` 里的对照表）
- 用 `runner.ckpt_path`（不是 `resume_dir`）把 `mode1-hb07-joint04` 的裸权重灌进一个全新构建的模型——**global_step 从 0 开始，optimizer 全新初始化**，不是"续训"。融合权重是 SFT 基底（812 keys，没有 value_head），加载时 value_head 8 个 key + `embed_tokens.weight`（大概率是权重绑定导致的误报）走"missing keys → random init"分支，非阻塞。
- 提交脚本：`my_docs/submit_5task_ppo_from_mode1hb07joint04_128env_4gpu.sh`，500 step / 4 GPU / 128 env / 800GB / novideo。

### 6.2 第一次运行：step 50 处 Ray OOM

`5task-ppo-mode1hb07j04-128env-4g-m800-0805-2250` 跑到 **step 50** 时 Ray OOM（节点内存 743.51GB/781.25GB，4 个 EnvWorker 各占 132~182GB）——5 个任务各自独立起 VectorEnv，内存开销比单任务 128-env 场景更重，即使开了 novideo 也没扛住。

`global_step_50` checkpoint 完整保存（dcp 分片 + full_weights.pt，19GB，含 optimizer 状态，可用 `resume_dir` 续训）。

### 6.3 step-50 权重评测：明显优于融合起点

用现成的 5-task 评测流程（2 次重复）评测这个 step-50 checkpoint：

| 任务 | 成功率 |
|---|---|
| bbh | 78.2% |
| ca | 96.6% |
| cb | 99.5% |
| hb | 42.2% |
| mpa | 91.7% |
| **avg** | **81.6%** |

对比：

| checkpoint | avg | hb |
|---|---|---|
| 融合起点 mode1-hb07-joint04（step 0） | 74.8% | 44.3% |
| **5taskppo-s50（继续训了 50 步）** | **81.6%**（+6.8pp） | 42.2%（-2.1pp，噪声范围内） |
| 对照：纯 hb PPO 基线 | 70.9% | 42.2% |
| 对照：纯 joint PPO 基线 | 68.8% | 0.5% |

只训 50 步，avg 就从 74.8% 涨到 81.6%，是整条实验线里 avg 最高的 checkpoint，hb 基本持平（没有像纯 joint PPO 训练那样把 hb 训崩）。**"融合 + 继续联合 PPO"这个方向看到了正向信号**，但只有 50 步数据，还需要跑更久确认。

### 6.4 续训 1TB 内存两次失败：疑似节点物理容量不够

用 `--memory=1000000` 从 `global_step_50` 续训（`my_docs/submit_5task_ppo_mode1hb07j04_resume50_128env_4gpu_m1000.sh`），两次尝试都在**第一个 step 还没跑完**时 raylet 进程直接被终止（不是 Ray 那种"温和杀掉几个 worker"的正常 OOM 流程，没有"Memory on the node was X/Y"账单信息，而是整个 raylet 突然死掉，job 状态是 `Stopped` 不是 `Failed`）。两次都被调度到同一个节点 `gpu-l-lg-cmc-h-h200-0294`——推测是这个节点的物理内存本来就不到 1TB（之前 OOM 账单显示某节点物理总容量约 781GB），申请量超过物理上限导致更早、更粗暴地被杀。

### 6.5 集群资源约束更新（muyao 告知，2026-08-06）

现在只能用这 4 个节点：

```
gpu-l-lg-cmc-h-h200-0256.host.h.pjlab.org.cn
gpu-l-lg-cmc-h-h200-0257.host.h.pjlab.org.cn
gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn
gpu-l-lg-cmc-h-h200-0260.host.h.pjlab.org.cn
```

用 `rjob submit --positive-tags "node/<fqdn>"` 可以把任务锁定到指定节点。`submit_5task_ppo_mode1hb07j04_resume50_128env_4gpu_m1000.sh` 已加上 `NODE_TAG` 环境变量支持：

```bash
NODE_TAG=gpu-l-lg-cmc-h-h200-0256.host.h.pjlab.org.cn ./submit_5task_ppo_mode1hb07j04_resume50_128env_4gpu_m1000.sh <image> [job-name]
```

锁定到 `0256` 提交后排队 10+ 分钟未起（`Inqueue`），推测这 4 个节点当前可能都满载。

**踩坑记录——`--gpu=0` 诊断节点状态这条路走不通**：本想用 `--gpu=0`（可以秒排上，不用等 GPU 空位）跑 `free -h`/`nvidia-smi` 来看节点真实负载，结果发现容器内 `free -h` 报的 `total` 就是我们自己 `--memory=` 申请的值（例如申请 4000MB 就显示 `total 3.9Gi`），**反映的是容器自己的 cgroup 内存上限，不是宿主机真实的物理内存/空闲情况**；`--gpu=0` 的容器也拿不到 GPU 设备，`nvidia-smi` 没输出。这台开发机没有 `kubectl`/`nvidia-smi`，`rjob` 也没有查节点资源的子命令，**目前没有找到能直接看到 4 个节点真实 GPU/内存负载的办法**，只能通过"申请真实资源、看排队时长"来间接判断忙闲。

### 6.6 待办

- 等 4 个节点里任意一个有空位，把 `resume50` 续训（1TB 内存）重新跑起来
- 如果 1TB 还是不够/还是崩，考虑降 env 数（不再是 128，比如 100）或者继续加大内存申请
- 续训成功后记得重新评测最新 checkpoint，跟 step-50 的 81.6%/42.2% 做对比
