# RoboTwin beat_block_hammer RECAP 4096-rollout 实验

2026-10-31，Asia/Shanghai。

用户最终要求立即停止旧小数据实验，再启动4096数据实验。旧运行已停止，`iter000/pipeline_status.json`记为`PAUSED_BY_USER`；保留完整critic1000–6000步checkpoint、原128条数据与日志。旧实验不会继续优势标注或策略训练。

## 新实验范围

根目录：`/data2/sjy/recap/beat_block_hammer/rollout4096_v2/`。

- 8张RTX4090、Ray64CPU；SFT40k采集策略不变，norm_stats SHA256仍为`4dd564ed776931a965616a7a21fd31044a4511486a747cda9a3621b949acc7ab`。
- **4096条自主训练轨迹＋256条独立验证轨迹，总采集4352条**。每个worker采544条，前32条验证、其余512条训练。每GPU2环境，16个并行环境。
- 原450条SFT训练示范和50条验证示范使用新meta目录，数据和视频以只读路径复用；新回报、优势sidecar不覆盖旧实验标签。
- 场景候选池独立生成4352个seed，间隔1024，避免复用原945个场景循环凑数；记录实际模拟器seed和请求seed，检查回退与跨分片唯一性。既有接口字段名`success_seeds`在这里仅表示候选场景池，未按采集策略成败筛选。
- critic从通用预训练视觉/语言模型及新价值模块开始，最多8000optimizer steps；policy从原SFT40k开始，最多3000steps。无LoRA，不从旧6000-step critic或PPO权重恢复。
- critic micro4/global128；policy micro1/global64。保持此前优化器配方，数据量不是官方充分性保证。

## 必要修复与新增证据

1. **Spearman并列排名**：使用平均排名，常量输入返回NaN并记录defined标志；不再产生常量预测得满分的结果。训练Spearman改为一次完整global batch128样本的排序相关，而非microbatch4相关系数的平均。
2. **完整验证**：移除正式验证的max_samples截断。除各分片和SFT指标外，汇集所有自主验证预测计算`rollout/*`指标，分别报告成功组、失败组MAE和价值偏差。
3. **采集内存**：LeRobot保存Parquet后将内存中的图像Arrow表替换为空表，保留元数据；三个episode写入/清缓存/重新读盘测试通过。旧实现确实持续拼接InMemoryTable/ConcatenationTable，数据写盘并不意味着缓存已释放。
4. **场景指令刷新**：原RoboTwin `SubEnv.reset`在新场景setup前调用create_instruction，且复用初始化时的episode_info_list。采集专用reset wrapper在native reset后读取当前task.get_info，刷新自然语言，并保留显式评测指令。没有修改外部RoboTwin仓库源文件。
5. **策略噪声seed**：移动到环境reset之后设置，避免模拟器setup_demo内torch.manual_seed覆盖已经记录的策略seed。
6. **安全边界**：每batch后检查系统MemAvailable，低于100GiB则保留已完成episode并停止；记录每worker `progress.json`中的quota、数量、峰值RSS、Arrow缓存字节数。阶段开始检查显存、CPU内存和至少300GiB数据盘空间。无自动重试或恢复旧实验。

旧数据只读审计：112条训练rollout里，51条指令明确含left/right arm/hand，其中33条与初始图像红色方块方位不一致；16条验证轨迹中有4条明确左右手文本，均匹配。这与原reset复用初始scene info的代码路径一致。故此前仅以“数据少/过拟合”解释critic偏差不充分；新实验必须重新采集。此审计是画面侧向启发式检查，完整结果在`/data2/sjy/logs/recap4096_20261031/old_instruction_image_side_audit.json`。

## 验证与阶段推进

- 修复后79项回归测试通过，包含遵守“critic后停止、不进入策略训练”的执行器回归，含并列排名、常量预测、跨轨迹排序反转、写盘后释放缓存/读回、场景指令更新、候选池4096/256划分和新的验收规则。
- 真实预检：32条采集轨迹、每worker两次batch重置；计算回报；8GPU critic2步训练、完整验证和checkpoint保存。全部通过才开始4352条正式采集。
- 本次实际执行：`collect → returns → value → 选择/验收critic → 停止`。遵守用户先前“不进入策略训练”的要求；`execution_plan.json`中`stop_after=value`，完成后状态为`PAUSED_AFTER_CRITIC`。优势、策略和最终对比评测命令已准备，但本次不会自动执行。
- critic每500步保存并完整验证。从已完整保存、满足验收的checkpoint里选自主验证MAE最低的一份；不固定使用最后8000步。
- 工程验收：自主验证整体MAE≤0.15、标准Spearman≥0.3且有定义；成功组和失败组MAE各≤0.2；SFT验证MAE≤0.05。以上不是论文规定的收敛阈值。没有checkpoint通过时停止，不进入优势标注和策略训练。
- 后续若用户要求继续策略阶段，再以同一96条评测manifest及可复现策略噪声对照SFT40k和更新策略。短跑成功率不作为正式提升结论。

## 文件与命令

- 生成静态配置：`examples/recap/prepare_robotwin4096.py`。
- 阶段入口：`bash examples/recap/run_robotwin4096.sh <stage>`。
- 有限流程：`.venv/bin/python examples/recap/run_robotwin4096_pipeline.py`。
- 静态配置：`examples/recap/config/robotwin_*4096*.yaml`。
- 状态：新根目录`pipeline_status.json`、`status.tsv`、`datasets/rollout/rank_*/progress.json`。
- 日志：新根目录`logs/`；预检证据`preflight_gate.json`；选定critic`selected_critic.json`。
- 修改前备份和回归日志：`/data2/sjy/logs/recap4096_20261031/`。

所有数据持续写入数据盘。按旧128条约3.4GiB线性估算，4096条训练数据约109GiB，另加256验证约7GiB；这是估算，不含HF缓存、checkpoint或失败率变化。不会全量解码到CPU内存。


## 2026-10-31正式启动

- 真实32条采集预检于12:20:19 CST正常结束：5947帧、26次成功，16条train/16条val，全部scene seed唯一。八个worker写盘后的Arrow缓存均为0字节；峰值RSS约16.7–16.8 GiB/worker（包含CPU模型初始化峰值，不是仅数据占用）。
- 回报计算预检正常退出；8GPU critic2步训练、全局128样本标准Spearman、完整验证及checkpoint保存于12:23:08 CST通过。
- 预检门禁`preflight_gate.json`为PASS，正式采集于12:23:24 CST开始。当前是采集阶段，不代表4352条已经收集完。
- 为遵守不进入策略训练的要求，监督器按`execution_plan.json`限定到critic后停止；替换监督器时保留同一个正式collect驱动PID813828，未重启采集。记录在`collection_handoff.json`及`logs/pipeline_follow_launcher.log`。
- 当前`~/.cache/huggingface/datasets`实际解析到`/data2/sjy/cache/huggingface/datasets`，大型Arrow缓存也在数据盘。系统盘剩余约95GiB，未将新数据写入系统盘。
