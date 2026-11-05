# beat_block_hammer PPO100 评测与 PPO200 重启记录

日期：2026-10-30（Asia/Shanghai）。机器：ZJU，8张RTX4090，Ray资源池64CPU。仓库：`/home/tanhe/sjy/RLinf-robotwin`。

## 已完成的旧实验与六任务评测

PPO100于00:20:29正常结束，退出码0；最终global_step100拥有完整DCP metadata、8个分片和8,529,334,029字节的full_weights.pt。最后一次训练内单任务验证75/96=78.125%。

随后在01:00:04完成六任务评测，退出码0。每任务96条、每条400步；manifest的seed/指令精确匹配，session/episode没有重复，所有指令非空。固定策略噪声seed为20261030。

- beat_block_hammer：77/96，80.2083%。
- place_phone_stand：19/96，19.7917%。
- pick_dual_bottles：48/96，50.0000%。
- lift_pot：4/96，4.1667%。
- move_can_pot：41/96，42.7083%。
- place_a2b_left：25/96，26.0417%。
- 总计214/576，37.1528%。

此前SFT40k同manifest单次结果分别为74、26、46、42、57、29条成功，总计274/576=47.5694%。这些差异是描述性比较：此前未固定相同的策略噪声，不能将差值全部归因于单一因素。lift_pot的下降很大，说明专门优化beat时，需要额外观察其他任务能力；修复PPO数学一致性不自动保证单任务训练保留所有任务性能。

结果：`/data2/sjy/eval/robotwin1_bbh_ppo100_eval6_20261030/summary.json`；日志：`/data2/sjy/logs/robotwin1_bbh_ppo100_eval6_20261030/eval.log`。

## 实际重算不一致及原因

首先运行真实8GPU/FSDP、从SFT40k开始、actor冻结的单步审计。它在第一次optimizer update之前被一致性检查拦住，01:14:21退出码1，未启动PPO200。

首批保存的有效数据共有8个样本、来自5个microbatch；单分量logprob最大差0.1565366，整段50×14维chunk logprob最大差0.7314746，对应概率比范围0.7233–2.0781。value最大差0.0010042。chain输入在模型内外均为FP32；本实验FSDP mixed_precision三项均为null，所以不是FSDP根输入BF16转换造成的问题。

同一批数据、同一SFT40k权重，在不使用FSDP的标准模型中重放，eval/train两种模式都逐位匹配原FSDP重算结果，与rollout保存值仍不匹配。

定位到的路径差异：

- rollout：`input_transform -> precision_processor（移到GPU） -> Observation.from_dict（uint8归一化） -> 图像预处理`。
- 原训练重算：`input_transform -> Observation.from_dict/图像预处理（CPU） -> 移到GPU`。

CPU和GPU的FP32图像运算有舍入差异；256级灰度RGB测试的768个值中333个不同，最大差1.1920928955078125e-7。这种微小输入差异经过BF16视觉/语言网络后，会影响高维Gaussian chunk logprob。

在标准模型中保持权重与真实输入不变，仅将重算观测在归一化前移到GPU，5组有效microbatch的component和chunk logprob差异全部变为0。此对照建立了该路径差异与本次重算偏差之间的因果关系。

原始证据：`/data2/sjy/diagnostics/bbh_logprob_20261030/initial/{recompute_report.json,replay.json,probe_gpu_preprocessing.json}`。`probe.json`和`probe_restored.json`属于不可靠的meta快速构造尝试，未匹配标准模型，**不作为结论依据**。

## 正式代码改动

1. `rlinf/models/embodiment/openpi/openpi_action_model.py`：default_forward在Observation.from_dict之前调用precision_processor，与rollout的设备顺序相同。
2. `rlinf/workers/actor/fsdp_actor_worker.py`：可选保存第一次更新前的真实批次；一致性门限增加整段chunk logprob检查（max abs差>0.05时中止），保留component/value检查。
3. `rlinf/utils/metric_logger.py`：TensorBoard改为写入`<log_path>/<experiment_name>/tensorboard`，避免不同实验的相同tag/step混在一起。checkpoint目录不变。

已有critic预热切换修复继续使用：保留同一个optimizer和scheduler，保持critic的Adam状态，在边界解冻actor。

相关44项CPU测试全部通过，Ruff检查通过。新回归覆盖观测设备处理顺序、实验TensorBoard隔离、预热边界optimizer/scheduler与Adam状态保存；真实GPU验证使用实际rollout数据。源文件改动前备份在`/data2/sjy/diagnostics/bbh_logprob_20261030/source_before/`。

## 新实验配置

配置：`examples/embodiment/config/robotwin1_beat_block_hammer_A10040k_ppo200_w200_ep2_8gpu.yaml`。

- 从`/data2/sjy/checkpoints/robotwin8_A100_40k/full_weights.pt`重新开始；resume_dir为null，使用对应norm_stats。
- 单任务beat_block_hammer；8GPU，Ray64CPU；16训练环境，8验证环境。
- max_steps=200；每20step保存，每10step验证；每次验证96条轨迹。
- critic_warmup_steps=200，单位是optimizer update。update_epoch=2、global_batch128时每个PPOstep有8次update，因此前25个PPOstep只训练critic，第26步起更新actor。
- 每轮64条训练episode，每条最多8个50动作chunk；global_batch128、micro_batch2；actor lr5e-6，critic lr1e-4。
- rollout.eval_seed=20261030；actor.verify_rollout_consistency=true。

输出：`/data2/sjy/rl_runs/robotwin1_beat_block_hammer_A10040k_ppo200_w200_ep2_8gpu/`。日志：`/data2/sjy/logs/robotwin1_beat_block_hammer_A10040k_ppo200_w200_ep2_8gpu/`。训练启动器：`/data2/sjy/run_bbh200_guarded.sh train`。

正式启动前还必须满足修复后的真实8GPU短跑正常退出、整批重算检查PASS、完整短跑checkpoint及代码hash校验通过。状态门禁文件：`/data2/sjy/diagnostics/bbh_logprob_20261030/verified.json`；状态衔接日志：`/data2/sjy/logs/robotwin1_bbh_ppo200_gate_20261030/stage_gate.log`。

## 权重清理

按授权删除旧PPO100的global_step10、20、30、40、50、60、70、80、90，释放182,945,181,096字节（约183GB）。保留最终PPO100 step100、SFT40k初始模型、日志和诊断证据。清理清单：旧PPO100日志目录下`cleanup_intermediate_20261030.json`。

## 修复后的实际GPU门禁结果与提交

修复后的8GPU短跑于01:49:45正常退出（exit0），完成1个PPOstep、2次critic optimizer update和完整checkpoint保存。8个rank共保存64个microbatch，其中70个有效样本的component logprob、chunk logprob、value最大绝对差全部为0，概率比最小/最大均为1。一致性检查PASS后生成verified.json，包含源文件SHA256；新训练启动器核对hash一致后才允许启动。

正式PPO200于01:50:13启动，驱动PID2007381；当前处于模型初始化阶段。新TensorBoard事件独立写入正式实验目录，6019配置为分别显示PPO100和PPO200，可在相同SSH转发下刷新查看。

## 正式训练首轮确认

PPO200已于02:05:50完成首个PPOstep（TensorBoard index0），并继续执行后续训练。首轮optimizer_steps=8；actor/policy_loss=0、approx_kl=0、clip_fraction=0，符合前25个PPOstep只预热critic的配置。正式训练自身记录的component/chunk/value重算最大差也全部为0，确认修复在64条episode、4个rollout epoch的正式路径中生效。

额外清理已验证的诊断短跑step1 checkpoint，释放20,327,242,460字节，CPU批次快照、日志和PASS证据保留；verified.json已标注该诊断权重不再保留。连同旧PPO100的10–90step中间权重，本次共释放203,272,423,556字节（约203GB）。SFT40k初始权重和PPO100最终step100保留。

指标解读：当前actor/ratio类日志对microbatch直接平均，完全被mask的空microbatch会使日志均值偏低；它不等同于本次有效样本的真实概率比。验证报告中的有效样本比值均为1。现有诊断开关仅在第一次optimizer update前检查首个global minibatch，后续诊断字段为0不能当作每一步都重新检查的证据。

## 运行进展更新：2026-10-30 10:59:39 CST

本次按用户要求更新实验记录；训练保持原样运行。TensorBoard零起始index为89，对应已完成90/200个PPOstep。驱动进程PID2007381的状态：`2007381    09:09:26 Sl   /home/tanhe/sjy/RLinf-robotwin/.venv/bin/python -u examples/embodiment/train_embodied_agent.py --config-name robotwin1_beat_block_hammer_A10040k_ppo200_w200_ep2_8gpu`。

### 验证结果

每次验证96条轨迹。以下步数是实际完成的PPOstep（TensorBoard index加1），不是optimizer update次数：

- 第10步：81/96，84.3750%。
- 第20步：81/96，84.3750%。
- 第30步：70/96，72.9167%。
- 第40步：77/96，80.2083%。
- 第50步：84/96，87.5000%。
- 第60步：81/96，84.3750%。
- 第70步：83/96，86.4583%。
- 第80步：88/96，91.6667%。
- 第90步：78/96，81.2500%。

这些结果存在波动，不能据此断言成功率稳定单调上涨或模型已经收敛。训练集rollout成功率和固定验证集成功率分别记录，不混为同一指标。

### PPO与critic指标

- 训练rollout最近一轮成功率：68.7500%；最近20步均值72.3438%。
- actor approx_kl：最新0.037047，最近20步均值0.027861，范围0.008375–0.065458。
- actor clip_fraction：最新0.101562，最近20步均值0.110352，范围0.080078–0.152344。
- critic value_loss：最新0.060119，最近20步均值0.049396，范围0.035140–0.063945。
- rollout critic explained_variance：最新0.309581，最近20步均值0.376911，范围0.206587–0.494291。
- train critic explained_variance：最新-0.103159，最近20步均值0.081252，范围-0.103159–0.270529。

本实现监测的是状态价值V的拟合，并非独立Q-function的收敛。explained_variance仍有限，不能宣称critic已经收敛。前25个PPOstep的approx_kl和clip_fraction均为0，符合critic-only warmup；第26步以后actor参与更新。

### 上次问题的复查范围

- 真实8GPU修复验证与正式首批component/chunk logprob、value最大差均为0；细节、对照实验与44项测试见本文前面章节。
- 一致性诊断只在第一次optimizer update前检查首个global minibatch；后续字段为0不代表每轮均重新审计。当前没有新增证据表明旧图像预处理顺序偏差再次出现。
- 本次读取的标量中非有限值记录数：0。最近4MB训练日志中特定致命错误标记：`{}`。此为有界日志扫描，不等同于完整日志的所有异常均已排除。

### checkpoint与资源

- 已核对完整checkpoint步数：20、40、60、80；每个均有DCP `.metadata`、8个非空分片，以及8,529,334,029字节的`actor/model_state_dict/full_weights.pt`。检查覆盖文件结构和大小，未进行额外完整模型加载或哈希校验。
- checkpoint直接写入`/data2/sjy/rl_runs/robotwin1_beat_block_hammer_A10040k_ppo200_w200_ep2_8gpu/checkpoints/`，每20步保存一次。
- 系统内存和数据盘：

```text
total        used        free      shared  buff/cache   available
Mem:           503Gi       266Gi        43Gi        63Gi       193Gi       170Gi
Swap:          8.0Gi       8.0Gi       0.0Ki
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda         11T  6.6T  4.4T  60% /data2
```

- GPU快照（序号、型号、已用MiB、总MiB、利用率%）：

```text
0, NVIDIA GeForce RTX 4090, 13678, 24564, 0
1, NVIDIA GeForce RTX 4090, 20481, 24564, 0
2, NVIDIA GeForce RTX 4090, 13615, 24564, 0
3, NVIDIA GeForce RTX 4090, 14185, 24564, 15
4, NVIDIA GeForce RTX 4090, 13902, 24564, 58
5, NVIDIA GeForce RTX 4090, 13812, 24564, 0
6, NVIDIA GeForce RTX 4090, 14184, 24564, 0
7, NVIDIA GeForce RTX 4090, 14182, 24564, 0
```

swap已用满需要持续关注，但单次快照不能判断内存泄漏，也不能单凭swap占用判定训练失败。

后续按原配置继续至200步，每20步保存、每10步验证；本次只更新文档，没有重启训练或删除权重。

原始巡检快照：`/data2/sjy/logs/robotwin1_beat_block_hammer_A10040k_ppo200_w200_ep2_8gpu/snapshot_20261030_105939.json`。本次仓库HEAD：`8a9aeb189dc7a11970a536b94d28fd1629b40827`（工作区含未提交补丁，HEAD本身不能完整代表运行源码；代码修复SHA256证据仍以verified.json为准）。


## 用户停止本实验并转向RECAP：2026-10-30 20:06 CST

用户已要求暂停并删除旧PPO权重。本实验最后完整训练标量为179/200步，最后完整验证为第170步60/96（62.5%）。训练驱动与专属Ray全部退出，GPU释放；不要自动重启或续训。

本次删除12个旧PPO checkpoint，243,926,908,090 bytes，包括本实验20–160步（间隔20）、旧PPO100最终100、旧六任务PPO50与smoke1/2。前文关于这些checkpoint存在/保留的描述是当时的历史快照；截至本节已删除。SFT30k/40k/50k、base、norm_stats、数据集、日志及诊断保留。

新算法审查及迁移建议见[RECAP论文与实现审查](2026-10-30-recap-paper-implementation-audit.md)。清理清单见`/data2/sjy/logs/ppo_retired_20261030/deleted_ppo_checkpoints.json`；本次没有启动RECAP训练。
