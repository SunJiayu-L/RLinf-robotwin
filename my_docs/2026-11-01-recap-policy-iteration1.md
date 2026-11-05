# RoboTwin beat_block_hammer：RECAP 第一轮策略优化

## 本次授权与起点

2026-11-01 用户接受暂按当前 critic 训练正确处理，确认优势标注、条件策略训练、评测流程后要求“开始执行”。本次授权替代此前“不进入策略训练”的停止点。

critic 使用本实验的完整 7000-step checkpoint（独立 rollout 验证 MAE 最低），而非 8000-step 最后一个 checkpoint：

`/data2/sjy/recap/beat_block_hammer/rollout4096_v2/runs/value/checkpoints/global_step_7000/actor`

此前 0.15 MAE 是未经本任务标定的工程门槛，当前按用户决定绕过。保留原始验证指标、失败状态备份和旧质量检查函数，没有把历史结果改写成通过。`selected_critic.json` 记录选择依据和源文件 SHA256。

## 优势标注

输入只包含 450 条 SFT train 示范和 4096 条 rollout train 轨迹；50 条 SFT val、256 条 rollout val 不参与阈值或策略训练。

- 冻结 critic，8 GPU 并行推理。
- lookahead=50，gamma=1；奖励和价值均以 800 为归一化尺度。
- 每任务阈值，参考训练 rollout 的优势分布，top 40% 左右为正。
- `force_sft_positive=false`，SFT 同样按估计优势标注。
- 写入 `meta/advantages_robotwin_adv_4096_v2.parquet`，不覆盖原始 reward/return。
- 标注后检查每个 episode/frame 键完整且唯一、与 returns sidecar 一致、数值有限、正负标签与同一阈值一致；记录两类数据的正样本率与文件 SHA256 到 `advantage_report.json`。

## 条件策略训练

从 `/data2/sjy/checkpoints/robotwin8_A100_40k` 的原始 SFT40k 独立初始化，策略 step0 开始。不是从 critic 或之前 PPO checkpoint 恢复。

- 全量微调，不使用 LoRA；8 张 RTX4090，Ray 64 CPU / 8 GPU。
- micro batch=1/global batch=64，8 GPU 下每卡梯度累计8次。
- lr=5e-6，cosine scheduler，warmup=300，总更新3000步。
- 每500步保存；输出 `runs/policy/checkpoints/`。
- 正负优势条件都训练（`positive_only_conditional=false`），条件 dropout=0.3。
- SFT / rollout 按帧采样的期望比例为 1:1；这是实验选择，不是论文固定比例。
- 先在单独 `runs/policy_smoke/` 完成2步和完整 checkpoint 保存检查，再从 SFT40k 开始正式训练；不从 smoke 权重续训。

## 评测及停止点

正式3000步权重写完整后，顺序执行 SFT40k 基线、正条件新策略，各96条轨迹，沿用同一测试 seed manifest 和采集环境，guidance scale=1。结果位于 `eval_baseline/` 和 `eval_policy/`。

这是一轮有限实验，评测完成后停止；没有自动开启下一轮 critic 或追加数据采集。运行失败时保存失败 stage 和原因，不自动从头重试。

## 执行入口与日志

- 新有限监督器：`examples/recap/continue_robotwin4096_policy.py`。
- stage 入口：`examples/recap/run_robotwin4096.sh`。
- 2026-11-01 16:50:48 CST 启动；初始监督 PID=1839741。
- 主日志：`logs/policy_continuation.log`。
- 优势日志：`logs/adv.log`。
- 策略检查/正式日志：`logs/policy_smoke.log`、`logs/policy.log`。
- 状态：实验根目录 `pipeline_status.json`、`status.tsv`、`execution_plan.json`。
- TensorBoard：`runs/policy/tensorboard/`。

实验根目录：`/data2/sjy/recap/beat_block_hammer/rollout4096_v2/`。

提交前已完成 Bash 语法、Python 编译、Ruff lint/format、只读 preflight。复查无重复进程，8 GPU 利用率低且剩余显存足够；没有停止其他用户进程。进入下一阶段前仍执行资源检查和 checkpoint/标签检查。

## 2026-11-01 19:51 CST 巡检

优势标注于17:40完成，逐帧覆盖和阈值一致性检查通过。2步策略检查于17:49退出码0，DCP metadata、8分片和8,526,574,741字节的完整策略权重已保存。正式策略训练17:49启动，目前约175/3000步，尚未到第一个500步正式 checkpoint。

最近50次更新：loss均值0.003369、grad norm均值0.126201，正标签采样率0.200313，保留优势条件比例0.698438。当前约39.6秒/更新，余下训练约31小时，另需保存及评测时间。未见近期OOM或异常退出；可用内存约400GiB，vmstat未显示实时换页。

需要关注的标签分布：rollout 845855帧中40%为正，阈值0.014692660033702851；SFT 51248帧中只有9帧为正（0.01756%）。原因是本次按rollout确定正阈值，SFT估计优势多数接近0且不强制正标签。1:1混合采样后，整体正标签比例预期约20.01%，实际最近50步约20.03%。这是当前设置的结果，不能把rollout 40%正样本误解成整个训练batch有40%正样本；此轮未据此改变配置或停止训练。

## 2026-11-01 19:56 CST：按用户要求停止并重开

本轮已停止，不再续训或自动评测。用户要求将确认的专家SFT作为正条件、rollout保留critic标签，并从SFT40k的step0重新开始。新实验为 `/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v3`；详情见同目录文档 `2026-11-01-recap-sft-positive-restart.md`。旧日志、旧标签与旧输出均保留，避免将这轮与修正后的结果混合。
