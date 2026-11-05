# RECAP 官方配置核对（2026-10-31）

本次仅核对资料和当前配置，没有修改运行中的 rollout4096_v2 实验。当前流程仍为采集、returns、critic、验证选模后停止，不自动进入策略训练。

RLinf 官方快照：`d80cf00e95e545f579e33432e7d7f14e86bb39d7`。下述是该版本的默认示例，不代表所有任务的最优参数。

## 数据量与混合比例

RLinf 文档的 LIBERO-10 Task 0 示例使用 4096 条自主轨迹、专家示范和独立 held-out 数据；4096 不是论文规定的单任务最低数据量。[官方说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html#dataset)

当前 RoboTwin 计划为 450 条 SFT 训练示范和 4096 条自主训练轨迹；验证集另有 50 条示范、256 条自主轨迹。它们是轨迹数量目标，不是 batch 内采样比例。

当前 critic/待执行 policy 配置将 SFT 总采样概率设为 0.5，8 个 rollout shard 各为 0.0625，并关闭 balance。因此期望按状态/帧采样 50% SFT、50% rollout；不是每个 batch 严格各半，也不是轨迹数各半。此比例是本实验选择，没有查到论文或 RLinf 要求固定 1:1。

官方 mixture 实现开启 balance 后：

`P(dataset_i) = weight_i * len(dataset_i) / sum(weight_j * len(dataset_j))`

关闭后仅按 weight 归一化。这里长度是 dataset 样本数，当前训练数据对应帧/状态。因此官方示例 SFT、rollout 都写 weight=1，并开启 balance，实际按样本量混合，不保证各 50%。不能只打开当前 balance 而不同时重设各 shard 权重。[官方采样代码](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/rlinf/data/datasets/recap/common.py)

## Critic 架构与训练

官方模型默认 SigLIP2 SO400M、Gemma3 270M，critic expert 为 `gemma_1m`，201 个离散 value bins，范围 [-1,0]。当前使用相同 backbone 家族和 bins，但 expert 为 `gemma_100m`。expert 配置名不表示整个模型只有该参数量。容量差异值得做对照，尚不能据此断定旧实验失败原因。[官方模型配置](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/examples/offline_rl/config/model/recap_value_model.yaml)

当前与官方一致的 critic 优化参数：backbone lr=5e-5、value lr=1e-4、weight_decay=1e-10、constant scheduler、warmup=500。因此此前不能仅凭恒定学习率或很小 weight decay 就认定它们偏离官方或导致下降。

官方 micro/global batch=32/256、no_shard；当前为 4/128、full_shard，是本地显存适配。官方配置 optimizer.total_training_steps=8000，但 runner.max_steps=-1、max_epochs=30000，不能把 optimizer 字段直接理解为必定在 8000 次更新停止。当前显式 runner.max_steps=8000，每 500 步保存/验证，属于本实验控制。[官方 critic 训练配置](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/examples/offline_rl/config/recap_value_model_sft.yaml)

## 论文与官方策略示例的区别

π*0.6 论文后训练使用 N=50 的优势估计、30% 优势条件 dropout，通常按任务使约 40% rollout 样本为正。论文没有统一的示范/自主采样比例；还包含多任务 value 预训练和少量多模态 web 数据共训练，当前单任务实验没有复现这两部分。论文在初始目标任务 SFT 阶段固定正条件、对人工纠正动作强制正条件；这不能直接推出后续 RECAP 阶段每一帧旧 SFT 都应强制正标签。[论文 §V-C、V-D 与附录 A-F](https://arxiv.org/html/2511.14759v1)

RLinf 教程优势示例为 N=10、top 30% 正样本；当前为 N=50、top 40%，更接近论文后训练设定。当前额外限定阈值参考训练 rollout，不使用 held-out 数据定阈值。[官方优势说明](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html#step-3-compute-advantages)

官方 policy 示例：micro/global=32/512、lr=1e-5、cosine、warmup=5000、30000 步，unconditional_prob=0.1、positive_only_conditional=true。当前仅准备了 micro/global=1/64、lr=5e-6、warmup=300、3000 步、unconditional_prob=0.3、positive_only_conditional=false 的配置；尚未自动执行。这些是明显的工程调整，不能称为官方原样复现。false 模式保留正负条件，true 模式将负样本作为无优势条件训练，两者训练目标也有区别。[官方 policy 配置](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/examples/offline_rl/config/cfg_rl_openpi.yaml)

## 下一阶段的判断依据

先确保新数据指令与场景匹配，再看独立验证集的 value 校准、排序和成功/失败分组误差。1:1、expert 大小、batch 和训练长度应明确作为实验因素；优先同数据比较 1m 与 100m，避免一次修改多个因素后无法解释结果。现有 critic 质量门槛属于本实验的工程门槛，不是论文阈值。

当前配置入口：

- `examples/recap/config/robotwin_value_4096_v2.yaml`
- `examples/recap/config/robotwin_adv_4096_v2.yaml`
- `examples/recap/config/robotwin_cfg_4096_v2.yaml`

本地官方 YAML 与采样源码快照另存于工作区 `investigation_20261030/official_recap_config_review/`，`manifest.json` 记录精确版本和 URL。
