# π*0.6 / RECAP 论文与 RLinf-RoboTwin 实现审查

日期：2026-10-30（Asia/Shanghai）。审查仓库：`/home/tanhe/sjy/RLinf-robotwin`，origin 为 `SunJiayu-L/RLinf-robotwin`，HEAD `8a9aeb189dc7a11970a536b94d28fd1629b40827`；保留现有未提交修改。

结论：**可以在现有 Pi05 SFT 模型上采用 RECAP 算法。仓库已有主要实现，但当前配置和 RoboTwin 接口未达到可直接启动的状态。RLinf 的总体方法合理，不能将默认 LIBERO 例程视为 π*0.6 全部训练细节的复现，也不能保证换算法后成功率一定单调上升。** 本次完成阅读、静态审查和有界 CPU 检查，未启动 RECAP 训练、未修改算法源码。

## 1. 旧 PPO 训练停止与权重清理

已按用户要求停止 PPO200 训练及其专属 Ray session，最后完整训练标量对应 **179/200 PPO steps**。训练驱动、Ray及其子进程均已退出，GPU占用回落至每卡约8–15MiB；没有终止其他用户进程。

第170步为最后一次完整验证，60/96=62.5%；第80步最高88/96=91.6667%。最后20个训练step的 critic explained variance 均值约0.0273，rollout侧约0.2726；这些记录保留，不继续旧实验。

2026-10-30 20:06 CST写入USER_PAUSED标记。随后删除 `/data2/sjy/rl_runs/robotwin*ppo*/checkpoints/global_step_*` 中12个旧checkpoint，共 **243,926,908,090 bytes（243.93GB，约227.17GiB）**：PPO200的20/40/60/80/100/120/140/160、旧PPO100最终100、旧六任务PPO50、旧六任务smoke的1/2。删除前校验解析路径、目录边界、无内部软链接；删除后确认这些目录均不存在。

保留：原始pi05 base、SFT30k/40k/50k、对应norm_stats、数据集、训练/验证日志、代码补丁、诊断快照。新算法仍需要初始SFT模型，因此本次清理范围是旧PPO实验权重。数据盘清理后约4.6TB可用。

清理证据：`/data2/sjy/logs/ppo_retired_20261030/{pause_processes.json,USER_PAUSED,deleted_ppo_checkpoints.json}`。停止过程中的短暂残留进程最终均退出；以finalize复查及GPU释放结果为准。

## 2. 论文方法与当前 PPO 的区别

阅读：[π*0.6论文](https://arxiv.org/html/2511.14759v1)（正文IV/V、附录A-F）和[RLinf官方RECAP文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)。

RECAP将示范与自主执行数据汇总，先拟合独立的201-bin分布式状态价值V，再计算n-step优势；策略通过优势指示条件和flow-matching目标学习，推理时选择正优势条件。它采用“采集一批数据—离线训练—再采集”的迭代流程，不依赖PPO的新旧logprob概率比。失败轨迹也有训练用途，不能只保留成功轨迹。人工纠正可选。

论文按任务定义价值尺度和优势阈值；附录给出的常用目标正样本比例是预训练约30%、后训练约40%，部分任务另行调整。条件dropout为30%，训练包括正、负优势指示。正文关于百分位的措辞应结合附录理解：实现时明确“正样本目标比例”，避免将top30%误写为30th percentile。

采用该算法不等于获得π0.6架构或权重：这里可继续使用已有Pi05/PaliGemma模型，准确名称应为“Pi05 + RECAP”，不宣称复现π*0.6模型。RECAP仍依赖可靠的价值预测与足够的数据覆盖；旧PPO的critic不收敛问题不会因为更换算法自动消失。

## 3. 你的仓库已经有哪些实现

仓库历史存在 `3b41eba6 feat(embodied): add Pi0.6 RECAP pipeline (#1015)`，不是本次新增。

- 回报标注：`examples/recap/process/compute_returns.py`。
- 分布式价值学习：`examples/recap/value/train_value.py`、`rlinf/workers/sft/fsdp_value_sft_worker.py`、`rlinf/models/embodiment/value_model/`。
- 优势标注：`examples/recap/process/compute_advantages.py`，支持torchrun分片。
- 条件策略训练：`examples/recap/cfg/train_cfg.py`、`rlinf/workers/sft/fsdp_cfg_worker.py`、`rlinf/models/embodiment/openpi_cfg/`。
- 数据与tag sidecar：`rlinf/data/datasets/recap/`，`meta/returns_<tag>.parquet`、`meta/advantages_<tag>.parquet`。
- 模型注册已有`cfg_model`、`value_model`，不是给现有PPO配置换一个loss_type即可使用。

另检查官方main固定提交 `d80cf00e95e545f579e33432e7d7f14e86bb39d7` 的8个核心文件。官方已将回报/优势脚本迁移到 `examples/offline_rl/advantage_labeling/recap/`，并将价值模型迁移到 `rlinf/models/embodiment/value_model/recap/`。比较时按新路径匹配，没有将404误判为删除功能。

其中CFG策略模型、CFG dataset、分布式critic模型3个文件与本仓库逐字节一致，其余5个存在差异，主要需结合离线训练重构检查；这不表示其余5个都错误。官方当前优势脚本也仍使用统一阈值与同样的归一化表达式，因此不能指望直接更新main就解决以下全部适配问题。

## 4. 发现的问题及适用边界

### A. RoboTwin价值数据通路未接入——直接启动的阻塞点

`rlinf/data/datasets/recap/value_model.py:305`的`ValueDataset._build_transform`只支持LIBERO/Franka；传入`robotwin`会报Unknown robot type。`examples/recap/process/compute_advantages.py:184`的KEY_MAPPINGS及`:395`的build_obs也不支持RoboTwin；`value_model/checkpoint_utils.py:138`的推理transform再次限制环境类型。

需要贯通同一组ALOHA观测：cam_high、cam_left_wrist、cam_right_wrist、14维关节/夹爪状态及任务指令。现有`robotwin_aloha_dataconfig.py`和`aloha_policy.py`可复用；不能简单假装成7维LIBERO。当前CFG策略接收成对wrist_images，ALOHA变换会拆成左右腕图像，不能仅凭obs_processor没有extra_view键就断言它缺一台相机。

### B. SFT40k初始化有加载错权重的风险——启动前必须修正/验证

实际检查：`/data2/sjy/checkpoints/robotwin8_A100_40k/full_weights.pt`存在，而同目录`model.safetensors`软链接指向`../openpi/pi05_base_pytorch/model.safetensors`。

`rlinf/models/embodiment/openpi_cfg/__init__.py:44–68`只检查`model_state_dict/full_weights.pt`和`actor/model_state_dict/full_weights.pt`，然后加载根部safetensors；它没有读取根部`full_weights.pt`的分支。因此直接把CFG的model_path设为现有40k目录且没有其他显式恢复操作时，会落入base软链接。

应增加明确的根部pt加载分支，或创建独立CFG初始化目录并提供其期望的结构；加载后记录所选真实路径、missing/unexpected keys，并抽样/完整比对模型参数。不要覆盖原SFT/base文件，也不要把`strict=False`无报错当作加载正确的证明。本文未把该加载器误用于新训练。

### C. 多任务统一优势阈值与论文不同

`compute_advantages.py:1172`将全部dataset/frame优势混合后取一个分位数，未按task_index/任务身份分组。尺度不同的任务可能几乎没有正样本；即使每个数据集各一个任务，最后也仍被合并。

有界合成检查中，两任务各4个样本，统一top30%阈值使A得到3个正样本、B得到0个。这证明存在这种机制，不代表已测得真实RoboTwin数据存在相同数值偏差。

建议按稳定的任务身份计算阈值并保存每任务正样本比例；跨数据集task_index可能重用，不能只按裸整数合并。采样权重/任务长度也应纳入报告。单任务没有跨任务问题，但阈值相等时`>=`可能导致实际正样本比例超过目标，需要记录ties。

### D. 归一化须统一；当前非默认参数可复现尺度不一致

`ReturnNormalizer.normalize_value`使用`return/abs(return_min)`；优势脚本`:592`使用`(x-return_min)/(return_max-return_min)-1`。当return_max=0时两式相等，默认成功终止奖励为0，**不能据此宣称默认配置已发生数值错误**。

当return_min=-100、return_max=-10、输入=-10时，实际函数分别输出-0.1与0，已在CPU检查复现。若采用逐任务尺度或非默认奖励，必须共享同一线性缩放函数；累加奖励应除以尺度，不能随意施加仿射偏移。尺度应记录在checkpoint/sidecar中，训练、优势标注和重采集迭代保持一致。动作/状态norm_stats与价值return尺度是两件事，可以保留原8任务动作统计，同时另外定义价值尺度。

### E. 正条件分支是一个变体；配置可支持更贴近论文的训练

默认`examples/recap/cfg/config/libero_cfg_openpi.yaml:66–69`设置`positive_only_conditional=true`、`unconditional_prob=0.1`。实际路由函数把负样本全部送到无优势条件分支，仍参与训练，并没有把失败数据全部丢弃；但没有训练显式负优势条件。这是合理的单侧变体，不能称为论文完整配置。

更贴近论文的起点：`positive_only_conditional=false`、`unconditional_prob=0.3`、`guidance_type=positive`、`cfgrl_guidance_scale=1.0`。路由函数已支持正/负条件及dropout，CPU检查通过。不要未经验证直接增大guidance scale，较强CFG并非必然提升任务成功率。

### F. 现有PPO episode日志不是可直接训练的离线数据集

检查当前`episodes/train.worker-0.jsonl`包含task/seed/instruction/success/success_step等结果元数据，没有每帧RGB、关节状态和实际执行动作。这些结果不能直接转换成完整RECAP训练样本；也不能从seed反推当时随机策略的全部动作。

应使用SFT40k重新采集成功和失败的完整观测—动作轨迹，保存为LeRobot v2.1并绑定策略版本、原始seed、指令、success、终止/截断原因和实际执行动作。帧数、fps、action_horizon和奖励步长使用同一时间单位。400 env steps、50动作chunk不能被误记成只有8帧；n-step lookahead需要按保存数据的帧定义。

## 5. 合理的部分和已完成验证

下列核心设计与RECAP方向一致：成功终点0/失败终点负罚/其他步-1的MC回报；201 bins的分类价值分布；n-step优势的终点bootstrap置0；带优势prompt的flow matching；推理正条件与CFG接口。新的独立价值模型使用SigLIP2和Gemma3，不能直接拿旧PPO scalar value head充当已训练RECAP模型。

运行原始源码函数的CPU检查，没有修改算法实现：

- 4帧成功轨迹回报[-3,-2,-1,0]、失败轨迹[-303,-302,-301,-300]符合配置。
- 正样本单侧条件与正/负双侧条件两种路由均符合其各自逻辑。
- RoboTwin build_obs拒绝未知robot_type的阻塞可复现。
- 非默认return_max归一化差异和跨任务统一阈值的合成反例可复现。

证据：`/data2/sjy/logs/ppo_retired_20261030/recap_math_checks.json`。这些是有界源码函数检查，**不等于端到端GPU训练已通过**。本次没有训练价值模型、计算真实数据优势或证明RECAP能提升成功率。

环境导入复查：Torch `2.6.0+cu124`、Transformers `4.53.2`，`ValueCriticModel`和`OpenPi0ForCFGActionPrediction`均可导入。过程中出现可选TensorFlow/NumPy C-API警告，但该次检查最终正常退出；没有为此升级或替换训练环境依赖。仍需实际权重加载与GPU短跑验证。

## 6. 下一阶段建议

1. 保留旧PPO停止状态；以现有SFT40k作为新RECAP策略起点，先解决加载分支并核对参数。
2. 先打通beat_block_hammer单任务：三摄像头、14维ALOHA、动作horizon50、相同原动作norm_stats。下载/指定独立价值模型的SigLIP2-so400m和Gemma3-270M权重，不混用Pi05 policy权重。
3. 将该任务500条示范与新采集的完整成功/失败rollout组成训练混合；按episode/seed划分训练和验证，防止同轨迹不同帧泄漏。起始采集量可先256条做工程验证，这不是论文保证充分的数据量。
4. 用同一任务尺度计算return，训练201-bin critic；在独立成功/失败轨迹上检查MAE、排序相关性、完成时间趋势和失败辨识，不能只看训练loss。
5. 计算与执行动作窗口匹配的优势，保存独立tag并检查标签覆盖与比例；进行正/负条件flow matching。优先guidance scale1，用短跑确认所有样本、梯度和checkpoint，再进行正式训练。
6. 按固定96条评测协议与SFT40k比较；确认单任务链路可靠后接入六任务、逐任务尺度/阈值与均衡采样，再迭代采集。8GPU可以分阶段复用，无需同时常驻完整policy和value模型。

实施时只移植必要改动，不直接覆盖工作区或整仓升级。当前已有的RoboTwin指令、episode去重、图像预处理及TensorBoard隔离修复应保留。

## 7. 来源与版本证据

- 论文：[arXiv 2511.14759v1](https://arxiv.org/html/2511.14759v1)，[PI官方PDF](https://www.physicalintelligence.company/download/pistar06.pdf)。
- 官方方法说明：[RLinf RECAP](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html)。
- 固定提交的[官方优势脚本](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py)及[条件策略实现](https://github.com/RLinf/RLinf/blob/d80cf00e95e545f579e33432e7d7f14e86bb39d7/rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py)。
- 本次对比清单含各文件SHA256：`my_docs/recap-audit-20261030/recap_upstream_comparison.json`；CPU检查与清理清单也存放于该目录。
