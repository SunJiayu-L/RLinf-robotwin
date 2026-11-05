# RoboTwin Pi05 + RECAP 工程复现

日期：2026-10-30，Asia/Shanghai。远程仓库 `/home/tanhe/sjy/RLinf-robotwin`，原HEAD `8a9aeb189dc7a11970a536b94d28fd1629b40827`，保留所有原有未提交修改。该实验使用8张RTX4090及Ray 64CPU资源池，未恢复旧PPO。

## 方法范围

依据[π*0.6论文IV/V及附录A-F](https://arxiv.org/html/2511.14759v1)和[RLinf RECAP框架](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)，在现有Pi05 SFT40k上实现迭代离线RL：完整经验采集→MC回报→独立分布式价值模型→n-step优势→优势条件化flow matching→固定场景评测。

这是RoboTwin上的**Pi05＋RECAP算法适配**，不是π0.6架构/大规模预训练数据/论文实物机器人结果的完整复现。首先用beat_block_hammer验证闭环。critic是201-bin价值分布，不是沿用旧PPO的scalar value head；策略使用原始Pi05骨干，无LoRA。

## SFT正样本规则：保留两种模式

SFT全部设为正，是合理的专家先验，原RLinf的这一覆盖规则属于设计选择，不能简单叫程序错误。

论文优势标签的条件是 `A_t > epsilon_task`，并不等价于“episode最终成功”，也不只是`A_t > 0`。附录A-F说明预训练约30%的示范数据为正优势；后训练通常依据rollout优势分布选择阈值，使约40%为正。阈值按实际任务分组，而非把不同语言指令当作不同任务。

本次新增`advantage.force_sft_positive`：默认保留RLinf专家全正规则，设置false则按计算优势标注SFT；本次paper模式配置显式false。两种模式均有测试。推理采用positive条件，训练使用正/负指示并以0.3概率drop掉优势条件，guidance scale1。负标签仍用于训练，不等于删除失败数据或认定专家示范无效。

GPU短跑中912帧得到365个正标签，其余负标签仅用于验证数据通路；2步critic没有达到可靠打分能力，不据此判断示范质量。

## 数据与奖励协议

根目录 `/data2/sjy/recap/beat_block_hammer/iter000/`。

- 原始LeRobot数据 `/data2/sjy/datasets/lerobot/robotwin_demo_clean_aloha_agilex_23task_v21`，beat_block_hammer为原始episode 0–499。
- `datasets/sft_train`：0–449，共450条；`datasets/sft_val`：450–499，共50条。目录链接原parquet/video，只复制选择元数据；不改原始数据，不重编号原episode，不重算动作norm_stats。
- `datasets/sft_smoke`：前8条，912帧，仅用于流程短跑。
- 正式采集计划128条，8个RLinf collector worker，每worker2环境、16条；其中每worker2条保留验证，共112条训练、16条验证。全部保留成功和失败轨迹。
- 采集保存每个实际执行动作对应的三路RGB、14维物理关节/夹爪状态、14维实际动作及指令；每帧独立记录，不将50动作chunk误存为一帧。LeRobot逻辑fps15，执行chunk50，上限400控制步；不将其当作实物机器人的实测控制频率。
- 常规帧奖励−1，成功终点0，失败终点−400；此任务return按固定800缩放，确保失败回报位于201-bin支持范围内。该数值是RoboTwin协议选择，不是声称论文规定−400/800。
- n-step为50、gamma1。原8任务动作/状态norm_stats保持不变；价值回报尺度与动作norm_stats是不同的配置。
- demo/rollout采样概率各0.5，rollout的0.5再平均分给8个worker shard。源数据条目不是按文件长度决定概率。
- train/val均按完整轨迹分开；采集验收检查scene seed不重复、帧连续、动作和状态有限且均为14维。

## 已适配的代码

1. `rlinf/data/datasets/recap/robotwin.py`：统一三摄像头映射、RGB uint8与视频float[0,1]读取、任务级阈值、无仿射偏移的return/reward共同缩放。
2. `ValueDataset`、优势推理及价值checkpoint transforms：支持robotwin/aloha；显式传入真实episode IDs，避免稀疏验证集被LeRobot误认为缺本地文件并访问Hub。
3. `openpi_cfg/checkpoint.py`及factory：优先读取40k目录根部full_weights.pt，严格核对键；不会因model.safetensors软链接回退到base。短跑8个rank均确认813个tensor、missing/unexpected均为空。
4. `compute_returns.py`：正确读取`[False]`成功标记，避免Python `bool([False])`误判成功。
5. `compute_advantages.py`：任务级阈值；正式后训练以rollout训练数据为阈值参考，排除验证数据；支持明确的SFT专家先验覆盖开关。
6. `FSDPCfgWorker`：遵守配置中的梯度检查点开关，避免训练循环无条件关闭。
7. `ValueHead.get_cls_embedding`：CLS索引跟随当前观测设备；不根据已分片/卸载的参数视图创建CPU索引。真实8GPU训练后验证曾复现设备冲突，修复后通过。
8. `FSDPValueSftWorker`：按条目使用同一任务尺度；保留原分类bin MAE，同时新增连续期望值MAE。验证结果写入`value_validation.jsonl`，以便训练阶段间验收。

## 新增执行文件与静态配置

均在仓库`examples/recap/`：

- `prepare_robotwin.py`：构建不改源数据的选择视图。
- `configure_robotwin.py`、`finalize_robotwin_configs.py`：生成并明确paper模式配置。配置已保存到`examples/recap/config/robotwin_*.yaml`，不是仅在聊天中给参数。
- `collect_robotwin.py`：使用RLinf Cluster/Worker/ComponentPlacement与现有RoboTwinEnv采集，不另造训练框架。每个worker写独立LeRobot shard与策略/seed元数据。
- `validate_robotwin_data.py`、`validate_robotwin_rollouts.py`：真实视频/图片读取、完整数据与拆分检查。
- `run_robotwin_recap.sh <stage>`：单阶段入口，独立短Ray路径、flock防重、64CPU/8GPU及资源检查，异常后只清理本次session。
- `check_robotwin_gate.py`：校验短跑checkpoint、正负标签、实际权重变化和rollout结果，并保存代码SHA256。
- `run_robotwin_pipeline.py`：有限阶段执行器；任一步失败停止，不自动重启旧PPO，也不跳过失败阶段冒充成功。

阶段名称：`returns_smoke`、`value_smoke`、`adv_smoke`、`cfg_smoke`、`cfg_batchcheck`、`collect_smoke`、`eval_smoke`；正式阶段为`collect`、`returns_iter000`、`value_iter000`、`adv_iter000`、`cfg_iter000`、`eval_baseline`、`eval_iter000`。

## 首轮训练预算与验收

正式计划：critic最多8000个optimizer steps，global batch128、micro batch4；policy最多3000个optimizer steps，global batch64、micro batch1。策略全参数训练、梯度检查点开启，lr5e-6、warmup300。参数均在静态配置中，可审查。

critic先在独立验证数据上检查：连续价值MAE不高于0.15、Spearman不低于0.3，方可继续策略训练。这是本工程设定的初步验收门槛，**不是论文给出的收敛保证**；未通过则流程停止并保留诊断，不拿不可靠critic盲目更新策略。

最终使用同一96条scene/instruction manifest、相同推理步数和按episode重置的策略噪声seed，对比SFT40k与新策略。只有完整评测后才能报告成功率变化。2步短跑不能证明算法收敛或效果提升。

## 当前已验证与待完成

- Gemma3-270M、SigLIP2-so400m官方权重已下载至`/data2/sjy/recap_assets/`。
- 真正的train/heldout视频样本通过critic collator：3路 `[B,3,224,224]`，指令非空、目标回报有限。
- 8GPU价值训练2步、验证与DCP/full_weights保存已正常退出。
- 8GPU条件策略训练2步与DCP/full_weights保存已正常退出；正、负、无优势条件分支均参与。
- 回归包括RECAP、此前PPO/指令/图像预处理、manifest与seed分区相关测试；最终数量以`regressions_final.log`为准。
- 完整采集短跑已通过：8 GPU、16条轨迹、3621帧、11次成功，seed无重复。该小样本只验证采集流程，不表示RECAP效果提升。
- 采集首轮失败：collector把seed_offset当作环境数量偏移，部分rank种子切片为空；改为`seed_offset=rank`。
- 采集第二轮失败：Pi05的`TokenizePrompt(discrete_state_input=True)`需要state，CFG正负提示单独分词时漏传。已在同一输入变换位置生成三组token，共用归一化后、padding前的state；第三轮实际采集通过，并有回归防止预处理顺序再次不一致。
- 完整manifest检查发现collector在第一次reset前多推进了一次seed；已修正为使用构造时选定的首个episode，后续batch才推进。验证器核对heldout seed与指令配对。
- micro2/global64策略压力短跑已完成2步与checkpoint保存，退出码0；正式用micro1/global64。价值micro4/global128和新策略实际rollout检查继续由有限preflight执行器完成；全部通过才写PASS门禁并进入正式迭代。

日志：`/data2/sjy/recap/beat_block_hammer/iter000/logs/`；阶段状态`status.tsv`；正式监督状态`pipeline_status.json`（正式提交后生成）。源码修改前备份：`/data2/sjy/logs/recap_adaptation_20261030/source_before/`。


## SFT正样本的两种定义

SFT成功示范全部作为正样本是合理的专家先验，不应简单称为代码错误。`force_sft_positive=true`保留RLinf原先默认行为；`false`对示范和自主采集都计算优势标签。本次以论文流程为目标的静态配置明确设为false，并保留开关用于后续对照实验。

论文的正标签是动作片段满足`A_t > epsilon_task`；后训练常用50步lookahead估计 `sum(r_t...r_{t+49}) + V(o_{t+50}) - V(o_t)`，终止后的bootstrap为0。成功轨迹中的停顿/纠错也不一定超过阈值；失败轨迹中的局部有效动作也可能超过阈值。阈值是相对排序标准，不要求固定等于0。论文附录给出预训练约30%示范、后训练通常约40%自主rollout为正的设定，不能理解为所有SFT示范天然都是论文定义的正优势。

2-step critic用于验证管线，并未收敛；短跑365/912为正是40%分位数标签流程的结果，不能用于评价其余示范好坏。正式critic通过独立验证门槛后才生成正式标签。


## 2026-10-31显存与正式batch选择

策略压力短跑使用micro2/global64，8张4090的显存观测约23508–23707 MiB（约23.0–23.2 GiB，占24 GiB卡约96%），接近24GiB容量。正式配方改用micro1/global64，通过梯度累积保持每次更新的有效batch64，保留更多运行余量。micro1训练与micro2/global64压力检查分别记录；不将不同采样随机序列下的梯度声明为逐位相同。


## 执行和状态查询

- 自动阶段衔接：`examples/recap/finish_robotwin_preflight.py`先等待策略压力短跑正常退出，接着价值batch检查、8条新策略rollout、验收门禁，再执行第一轮正式采集→回报→价值训练→优势标注→条件策略训练→成对96条评测。任一阶段失败则停止，不自动重启。
- 启动日志：`/data2/sjy/recap/beat_block_hammer/iter000/logs/preflight_controller.log`。
- 单阶段结果：`status.tsv`中的`START/END/exit`；完整运行状态：`pipeline_status.json`；验收证据：`smoke_gate.json`。
- 训练数据与权重、日志全部直接落`/data2/sjy/recap/`；旧PPO维持停止。原SFT40k和norm_stats保持不变。
- 该实现为Pi05+RoboTwin上的RECAP算法适配，不等同于复现论文Pi0.6大模型架构或实物机器人结果。优化器和混合参数dtype沿用本仓库RLinf路径；未额外实现FP32 master-weight优化器，不能声称数值配方与论文完全一致。


## 2026-10-31 00:31 CST 正式启动记录

- 最终回归：67项通过；Ruff检查通过。
- 价值micro4/global128两步检查及验证、8份DCP分片和full_weights保存通过；该短跑验证MAE约0.187、Spearman约−0.038，只证明代码运行，不代表critic可靠。
- 新条件策略实际rollout：8条完整轨迹、1510帧、7条成功，8个worker均正常退出。heldout seed和指令配对检查通过。这不是正式成功率对比。
- `smoke_gate.json`为PASS；确认2步策略权重实际改变（action_out_proj.weight最大绝对变化约4.2431e-6），并记录关键代码/静态配置SHA256。
- 2026-10-31 00:31:13 CST，正式`run_robotwin_pipeline.py`启动；00:31:23开始`collect`阶段。当前是正式采集阶段，尚未完成8000步critic或3000步策略训练。
- 正式采集128条，8个worker各16条，其中各2条验证；共112训练/16验证。采集池945个场景seed与96条正式评测manifest无交集。
- 后续顺序：`collect → returns_iter000 → value_iter000 → adv_iter000 → cfg_iter000 → eval_baseline → eval_iter000`。critic门槛未通过时停止，不跳过；旧PPO不恢复。
- 输出和日志根目录均为`/data2/sjy/recap/beat_block_hammer/iter000/`。`pipeline_status.json`给出正式运行状态，`status.tsv`提供阶段退出码，`logs/formal_pipeline_launcher.log`提供阶段执行器错误。
