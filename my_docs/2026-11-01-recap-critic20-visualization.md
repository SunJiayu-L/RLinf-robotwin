# Critic 7000-step：20 条验证轨迹可视化

## 采样与推理

- 实验：`/data2/sjy/recap/beat_block_hammer/rollout4096_v2`。
- 权重：`runs/value/checkpoints/global_step_7000/actor/model_state_dict/full_weights.pt`；按独立 rollout 验证 MAE 最低选择，未选择最容易展示的 checkpoint。
- 从验证集固定随机抽取：10 条 SFT 示范，7 条成功 rollout，3 条失败 rollout。抽样 seed=20261101，未按预测曲线好坏挑选。
- 推理只使用空闲 GPU7，batch=4，BF16 autocast、eval 模式、与训练验证相同的 ValueDataset/ValueDataCollator。严格加载全部权重；峰值保留显存 2064 MiB，退出码 0。
- 完整推理 3577 帧：SFT 1131 帧、成功 rollout 1246 帧、失败 rollout 1200 帧。
- 无额外训练、无优势阈值拟合、无正式正负标签写入；保持策略训练停止。

## 曲线含义

总览橙线为实际累计未来回报 `R_t/800`，蓝线为 critic 估计 `V(o_t)`。`R_t` 是该次执行的 Monte Carlo 实现值，不是可直接获知的真实条件期望。

逐条 PDF 另画真实单步 reward（原始单位）和 `A50`。失败 reward 图采用 symlog 纵轴，能同时显示通常的 -1 和末尾 -400。

`A50(t) = sum(r[t:min(t+50,L)])/800 + V(o[t+50]) - V(o[t])`；gamma=1；超出记录结束后 bootstrap 为 0。本实验超时按失败终止并包含 -400 惩罚，此处理不是所有 truncation 场景的通用规则。

前 50 帧可以比较，图中用浅灰底标出，没有丢弃。优势使用未来 50 步，并不需要之前 50 步历史。最后 50 帧需要终止处理，优势图用另一块灰色尾部标出。

起始状态未来结果具有不确定性，不能因某条轨迹最后失败就要求 critic 在起点精确预测这次失败回报；比较时应结合分布校准、后续事件和优势变化。删掉前 50 帧也不能消除所有偏差，例如 ROL-OK-02 在第 50 帧之后的 MAE 仍为 0.50586。

## 此次抽样结果

按所选帧加权 MAE：SFT 0.001283；成功 rollout 0.267889；失败 rollout 0.212020。它们仅描述本次固定抽样，不能替代完整验证集指标；长轨迹会对帧加权指标贡献更大。成功 rollout 按轨迹等权平均 MAE 为 0.220350。

## 验证与交付

- 每条实际 reward 反向累计后都与 returns sidecar 一致。
- 推理 target 与对应 sidecar 完全对齐（误差小于 1e-6），逐条 episode/frame 索引和帧数校验通过。
- 用实际累计回报替代 V 后，50 步回报重构残差小于 1e-10，检查切片及终止边界。
- 没有平滑曲线；SFT 与 rollout 总览纵轴范围不同，避免 SFT 细微误差不可见。
- 20 页 PDF 已逐页渲染检查；总览、详情、CSV、选择 manifest 和原始预测 JSON 一并保存。

远程分析目录：`/data2/sjy/recap/beat_block_hammer/rollout4096_v2/analysis/critic7000_20episodes/`。

关键文件：`sft10_value_vs_return.png`、`rollout10_value_vs_return.png`、`rollout10_advantage50.png`、`critic7000_20episodes_detail.pdf`、`selection_manifest.json`、`report.json`、`summary.json` 和每条轨迹的 CSV。

推理脚本：`/data2/sjy/logs/visualize_critic20_20261101.py`；日志同目录同名 `.log`。
