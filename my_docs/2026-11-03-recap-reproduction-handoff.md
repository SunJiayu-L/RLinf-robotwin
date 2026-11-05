# RoboTwin `beat_block_hammer` RECAP 复现流程与交接

更新：2026-11-03（Asia/Shanghai）。这是当前实验的交接文档，后续可直接修改。远程仓库 `/home/tanhe/sjy/RLinf-robotwin`；本地镜像目录 `C:/Users/sun/Documents/ChatGPT/weight merge/recap_adaptation`。远程数据根目录 `/data2/sjy/recap/beat_block_hammer/`。

## 1. 目标与复现范围

用 RoboTwin 的 `beat_block_hammer` 任务，将八任务 SFT40k 策略作为初始策略，采集自主轨迹，训练价值模型，按优势给自主轨迹标注正负条件，并用高质量 SFT 示范作为正条件训练 π₀.₅ 策略。当前只做单任务、单轮实验。方法参照 [π*₀.₆ 论文](https://arxiv.org/html/2511.14759v1)和 [RLinf RECAP 文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)。本项目使用 π₀.₅，且没有复现论文的多任务价值预训练、网页多模态共训练、人工在线纠正或 π₀.₆ 架构。

```text
SFT40k → 自主采集 → 计算回报 → 训练 critic → 估计优势/标注
                                     ↓
                 SFT专家正条件 + rollout优势标签 → 从SFT40k训练CFG策略 → 评测
```

“策略训练一步”是一个 global batch 的一次参数更新。当前不会每若干步重新 rollout；若要第二轮，需用新策略重新采集、并入历史数据、重新训练 critic/标注，再从基础策略权重开始新一轮。论文 Algorithm 1 是外层迭代，不指定每多少梯度步采一次。

## 2. 数据和模型

- 初始策略：`/data2/sjy/checkpoints/robotwin8_A100_40k/full_weights.pt`；对应统计：`/data2/sjy/checkpoints/robotwin8_A100_40k/physical-intelligence/robotwin/norm_stats.json`。SFT40k 来自 `beat_block_hammer`、`place_phone_stand`、`pick_dual_bottles`、`lift_pot`、`move_can_pot`、`place_a2b_left`、`place_empty_cup`、`handover_mic` 八任务联合 SFT。对 `beat_block_hammer` 的既有测评为84/96成功。
- 本任务 SFT：450条训练示范、50条验证示范；训练示范共51,248帧。它们来自先前整理的专家数据。已核对数据元信息和成功标志，但**未逐一人工复看450条视频**。
- 自主采集：4096条训练轨迹、256条独立验证轨迹，共4352条；训练845,855帧，验证49,999帧。训练轨迹3066/4096成功，验证201/256成功。使用独立场景seed池，记录请求与实际seed。原场景指令刷新的bug已在采集wrapper中修复，保证文本与新场景对应。
- 旧阶段数据及 critic：`/data2/sjy/recap/beat_block_hammer/rollout4096_v2/`。正式训练数据在其 `datasets/sft_train/` 和 `datasets/rollout/rank_00..07/train/`；验证数据与训练分开。新策略实验直接引用这些数据，不复制RGB视频。

## 3. 回报、价值与优势

回报按论文式稀疏反馈构造：非终止步 `r=-1`，成功终止步 `r=0`，失败终止步 `r=-400`；`γ=1`，按800归一化到价值模型尺度。环境单步reward不直接当训练目标。`G_t` 是这条已发生轨迹从 `t` 到终点的累计回报；`V(o_t)` 是 critic 对未来回报的条件期望。单条失败轨迹的早期 `G_t` 很低，不表示同样画面下理论 `V` 必须同样低。

critic 是独立的视觉语言价值模型，输出201档价值分布及其均值。训练使用8张RTX4090、Ray 64 CPU，micro batch 4、global batch 128，总8000次更新，每500步验证/保存。使用完整独立验证集选模，**7000步**的 rollout 验证 MAE 最低，约0.1731，Spearman约0.6225。这个效果仍有价值校准及泛化风险；早期写的MAE≤0.15是本实验人工工程门槛，不是论文标准，不能声称critic完全达标。选中模型：`/data2/sjy/recap/beat_block_hammer/rollout4096_v2/runs/value/checkpoints/global_step_7000/actor/`。

优势按 `A_t = (Σ_{i=0}^{49} r_{t+i})/800 + V(o_{t+50}) - V(o_t)` 估计；终止附近按实际短窗口和终止bootstrap处理。rollout按训练集优势第60百分位阈值 **0.014692660033702851** 标正，恰有338,342/845,855帧（40%）为正。held-out验证集不参与定阈值。阶段脚本见 `examples/recap/process/compute_returns.py`、`compute_advantages.py`，配置见 `config/robotwin_returns_4096_v2.yaml`、`robotwin_value_4096_v2.yaml`、`robotwin_adv_4096_v2.yaml`。

## 4. 标签修正及当前策略实验

第一次给SFT和rollout共用rollout阈值时，SFT仅 **9/51,248帧**标正。原因是critic在专家训练轨迹上几乎拟合了Monte Carlo回报，SFT的优势基本围绕0，而rollout正阈值是0.01469。这会把绝大多数成功专家示范教成负条件。旧策略实验 `rollout4096_v2` 已暂停，不能视为合理对照。

用户随后决定将经过确认的SFT示范全部标正，rollout仍按critic优势阈值标注。新标签tag `robotwin_adv_4096_sftpos_v3`，新实验目录 `/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v3/`。仅SFT布尔标签变化；连续优势、reward、return不变，8个rollout标签文件的SHA256与旧版一致。SFT全正是本次明确的专家锚点假设，**不是论文要求所有后训练SFT帧无条件标正**。标签依据和哈希见新目录 `expert_positive_manifest.json`。

策略配置：`examples/recap/config/robotwin_cfg_4096_sftpos_v3.yaml`。从**原始SFT40k**严格加载，从策略step0开始，`runner.resume_dir=null`；全量微调，LoRA关闭，8张RTX4090、Ray 64 CPU。global batch64、micro batch1，即每卡累积8次；SFT与rollout按帧采样概率约1:1（SFT权重0.5，8个rollout shard各0.0625，`balance_dataset_weights=false`）。预期混合正标签70%；运行日志实测累计约70.25%。正负条件均保留，`positive_only_conditional=false`，优势条件dropout30%。学习率5e-6、warmup300步、cosine，总目标3000步、每500步保存。与RLinf官方示例的大batch/30k步不同，这些是本机显存与实验时间下的工程选择。

实验监督器 `examples/recap/continue_robotwin4096_sftpos.py`；阶段脚本 `examples/recap/run_robotwin4096_sftpos.sh`。脚本原计划策略3000步后，以同一固定96条评测seed分别评测SFT40k与正条件策略，CFG scale=1，然后停止。**2026-11-02用户要求立即暂停**：停止在日志step996；只保存到完整 `global_step_500`，未生成1000步权重；监督器和专属Ray进程已退出，后续自动评测没有启动。新根目录有 `USER_PAUSED`，`pipeline_status.json` 为 `PAUSED_BY_USER`，不能直接再次运行现有监督脚本。

当前保留策略权重：`/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v3/runs/policy/checkpoints/global_step_500/actor/model_state_dict/full_weights.pt`；完整DCP目录在其 `actor/dcp_checkpoint/`。曾有17个4096实验中间/冒烟checkpoint被清理，释放约162.94GiB；保留了基础SFT40k、选中的critic7000及新策略500。清单：新根目录 `checkpoint_cleanup_20261102.json`。旧数据、标签、日志和TensorBoard事件保留。**新策略尚无96条成功率结果，训练loss下降不等于任务成功率提高。**

## 5. 以后修改时的检查顺序

1. 明确要检验的变量：例如SFT全正假设、SFT/rollout比例、critic质量、阈值/正样本率、训练步数或指导强度。每次尽量只变一个因素，使用独立实验根目录和标签tag。
2. 若调整标签但不改连续优势，可用版本化sidecar重标，核对原文件及rollout哈希；无需重复采集或critic推理。若换critic或重新采集，必须重算相关优势并核对数据/模型配套。绝不改原SFT40k权重和norm_stats。
3. 启动前检查8卡、Ray/进程、锁、盘空间、配置的`resume_dir`、初始权重路径、采样比例和正负标签路由。旧根目录标记为暂停，不能误用其脚本/优化器。
4. 训练中读 `logs/policy.log` 和 `runs/policy/tensorboard/`，同时看标签比例、正/负条件loss、梯度、显存及完整checkpoint。过去首次500步DCP有 `.metadata`、8个分片及 `full_weights.pt`；仅有目录不算完整保存。
5. 对最终模型在同一固定96个seed上评测SFT40k与新策略，报告成功数、轨迹时长、失败类型和不确定性。比较增益前要确认正条件CFG参数、指令、norm_stats和模拟器一致。当前这一步尚未执行。
6. 如果做论文意义上的第二轮，先用第一轮策略**重新采集新自主轨迹**，并入先前示范与rollout，重新训练critic/估计优势，然后从预训练策略权重重新微调。论文没有规定每几百个优化步采一次。

相关详细记录：`my_docs/2026-10-31-recap4096-experiment.md`、`my_docs/2026-10-31-recap-official-config-review.md`、`my_docs/2026-11-01-recap-sft-positive-restart.md`、`my_docs/2026-11-01-recap-critic20-visualization.md`。这些旧记录保留了各时间点的决策，不要把其中“计划中/当时运行中”误当作当前状态。
## 2026-11-03：条件 dropout 与下一阶段

论文在策略训练时随机省略优势条件，以便同一模型学习有条件和无条件行为；当前 `robotwin_cfg_4096_sftpos_v3.yaml` 已设置 `actor.model.openpi.unconditional_prob=0.3`，且最近982步日志中的 `train/conditional_ratio≈0.703`，说明该机制已实际执行。用户提出将这一机制的专门改动/消融放到后续计划：目前不修改实现，不重启已暂停的实验，也不以“尚未实现”描述现状。

下一步应先用保留的策略500步完整权重和原始SFT40k，在同一批固定96个seed上完成可比测评：检查正条件推理配置与norm_stats、统计成功数和轨迹长度。此项不会训练或重新采集。看到500步是否优于基线后，再决定是否从SFT40k重新启动单独的策略训练实验、延长训练到预定3000步，或调整critic/标签；不要把暂停在996步但尚未保存的状态当作可用权重。完整一轮之后如需论文式第二轮，使用新策略采集轨迹，合并旧数据，重新训练critic和标注，再从预训练策略起点训练新策略。后续可在固定数据与评测条件下比较dropout=0与0.3，同时明确推理使用正条件或CFG；注意关闭dropout后不能直接照用依赖无条件分支的CFG设置。
## 2026-11-03：固定96场景的500步策略测评

评测目录：`/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v3/eval500_vs_sft40k/`。两组均在8GPU、相同96个场景与相同记录的策略噪声seed下完成，8个分片各12条；数据校验通过。SFT40k使用`no_guide`，500步策略使用正条件、CFG scale=1。基线首次测评成功82/96（85.42%），累计17,586帧；500步策略成功73/96（76.04%），累计18,557帧，下降9/96即9.38个百分点。成对结果：63条两者都成功、19条基线成功但500步失败、10条基线失败但500步成功、4条都失败。策略500测评首次因Hydra缺少`+`语法在任何轨迹生成前退出；只补跑策略500，第二次退出码0并通过96条校验。训练仍暂停，500步仍处于较早阶段，不能据此认定完整3000步效果。
## 2026-11-03：按用户要求重新从SFT40k训练到3000步

用户明确要求不从500步续训。新实验目录：`/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v4_restart/`。复制并复核原v3版标签清单；原始SFT全正与rollout40%正的sidecar数据不变。新配置 `examples/recap/config/robotwin_cfg_4096_sftpos_v4_restart.yaml` 保持原配置的batch64/micro1、全模型、warmup300、lr5e-6、3000步与每500步保存，`resume_dir=null`，logger改为新目录。新的监督与入口脚本 `continue_robotwin4096_sftpos_v4_restart.py`、`run_robotwin4096_sftpos_v4_restart.sh`；Ray独立端口7811及短临时路径。旧v3目录仍保持`PAUSED_BY_USER`和500步权重，未用作初始化。新实验预检、Python编译、Ruff、Bash语法通过；2026-11-03 13:38 CST启动监督进程PID64588，Ray64CPU/8GPU；已严格加载SFT40k完整权重813 tensors、无missing/unexpected，训练真实完成至少2步，学习率从0开始warmup，未见OOM。新流程预定3000步训练完成后用同一96场景顺序评测基线和最终策略；此时仍在训练中，不把计划当成已完成结果。
