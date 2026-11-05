# RECAP：SFT 正条件示范重开实验

## 用户决定与旧实验停止

用户于2026-11-01明确要求：暂停当前实验，将确认过的高质量SFT示范作为正条件，rollout继续使用critic优势标签，从SFT40k重新开始。

旧实验 `/data2/sjy/recap/beat_block_hammer/rollout4096_v2` 已于19:56 CST停止。先冻结监督器以阻止自动进入评测，终止本实验训练driver，清理其独立Ray会话 `/data2/sjy/rt/r4k_174929`。监督器、训练driver及该会话已退出；没有停止其他用户任务。旧目录保留 `USER_PAUSED`、`PAUSED_BY_USER` 状态、日志和旧标签；不再从这次策略权重续训。

## 标签修正与隔离

新实验根目录：`/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v3`。

复用450条SFT训练示范、4096条rollout训练轨迹及原始7000-step critic的连续优势，不重新运行GPU标注。验证集不参与训练或阈值拟合。

新标签版本 `robotwin_adv_4096_sftpos_v3`：

- SFT：51248帧全部正标签，旧版本只有9帧为正。只修改布尔条件标签；连续优势、当前/未来价值、reward、return、episode/frame键均保留不变。
- Rollout：338342/845855帧为正，仍为40%；8个新标签Parquet与旧版本逐份SHA256完全一致。
- 旧版本的标签Parquet没有修改；新标签使用独立文件名。mixture metadata增加新tag，不改旧tag。
- SFT正标签是明确的专家行为锚点，不代表重新计算的优势全部大于阈值。用户确认使用该示范集合；元数据和末尾reward核对为成功示范，不构成对450条视频逐条人工审查。

沿用1:1的SFT/rollout采样比例，因此预期混合正样本率为70%，负样本率为30%；条件dropout仍为30%。

详细逐文件校验记录：新根目录 `expert_positive_manifest.json`。

## 新训练配置

- 基础策略：`/data2/sjy/checkpoints/robotwin8_A100_40k`；完整权重严格加载。
- `runner.resume_dir=null`，从step0开始，不恢复旧策略、旧优化器或2步pilot。
- 全量微调、LoRA关闭；8张RTX4090，Ray64CPU/8GPU。
- micro batch=1，global batch=64，每卡梯度累计8次。
- lr=5e-6，warmup=300，cosine，共3000次更新，每500步保存。
- `positive_only_conditional=false`，正负条件都训练，优势条件dropout=0.3。
- Norm stats沿用SFT40k，SHA256=`4dd564ed776931a965616a7a21fd31044a4511486a747cda9a3621b949acc7ab`，未修改。

配置为 `examples/recap/config/robotwin_cfg_4096_sftpos_v3.yaml`。采集/评测配置为 `robotwin_collect_4096_sftpos_v3.yaml`；优势配置记录 `force_sft_positive=true`，但本次已经复用连续估计完成重标，无额外critic推理。

本次只改变示范条件和实验输出隔离，训练引擎沿用前次已通过的2步运行与checkpoint检查。新正式训练启动后检查实际步数、loss有限性、标签路由和GPU内存。

## 执行与停止点

监督器 `examples/recap/continue_robotwin4096_sftpos.py`，stage入口 `examples/recap/run_robotwin4096_sftpos.sh`。新Ray端口7798，短临时目录前缀 `/data2/sjy/rt/r4s_`，与旧会话隔离。

2026-11-01 20:01:19 CST启动监督器，初始PID2368069；20:01:29启动策略阶段。启动前Bash语法、Python编译、Ruff及只读preflight通过，检查新旧标签hash、基础模型、无resume、GPU与系统内存。

正式训练3000步后，依次评测SFT40k基线和新策略各96条轨迹；新策略使用正优势条件，guidance scale=1；然后停止。出现失败只记录并停止，不自动重复启动。

关键位置（均在新实验根目录内）：

- 状态：`pipeline_status.json`、`execution_plan.json`、`status.tsv`。
- 日志：`logs/policy.log`、`logs/pipeline_launcher.log`。
- TensorBoard：`runs/policy/tensorboard/`。
- 正式权重：`runs/policy/checkpoints/`。
- 评测：`eval_baseline/`、`eval_policy/`。

## 20:12 CST 实测启动验证

已实际完成6次更新。日志确认严格加载SFT40k完整权重（813个tensor，无missing/unexpected keys），新run从step0开始，学习率重新从0 warmup；没有恢复旧优化器。

前6批正标签比例均值约68.75%，与70%的期望一致；各批loss与grad norm均为有限值，最新loss约0.00229。近期未见CUDA OOM，8个GPU worker持续运行。详细证据保存在新实验根目录 `restart_validation.json`。


## 2026-11-02 暂停与权重清理

用户要求立即暂停。本轮最后记录 step996；完整策略 checkpoint 为 step500，未生成 step1000。策略、监督器及专属 Ray 会话均已停止，后续评测不启动。

保留 SFT40k 原始权重、critic7000 完整 checkpoint、本轮 policy500 完整 checkpoint（含优化器分片）。清理其余4096实验中间和冒烟 checkpoint 共17个，释放约162.94 GiB。数据、标签、TensorBoard及日志保留。清单：`/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v3/checkpoint_cleanup_20261102.json`。
