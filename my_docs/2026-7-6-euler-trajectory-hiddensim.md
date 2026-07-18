# Euler 轨迹 HiddenSim 分析

**日期**：2026-07-06（本文档为修正版，取代同名旧版本）
**脚本**：`scripts/plot_euler_trajectory_heatmaps.py`
**SLURM job**：10204865（全量 10 对，成功）；冒烟测试 10204755/10204846（单对，用于验证新流程）
**输出目录**：`results/euler_trajectory_heatmaps_v2/`（31 个文件：10 对 × 2 [png+npz] + 5 单任务 × 2 [png+npz] + 1 个 `task_vector_cosine.json`）

---

## 0. 本次修正说明

旧版本文档存在多处方法论问题（经审阅确认），核心是：**用跨任务共享的随机像素/随机关节角+写死的 "beat block hammer" 指令**作为所有任务、所有 checkpoint 的探测输入，导致结果实际反映的是"随机合成输入下各 checkpoint 自身生成轨迹之间的隐藏状态差异"，而不是训练分布上的真实表征漂移；此外 merged 模型只用单一（且错误)观测代表两个任务、`load_weights` 遇到缺失文件会静默返回而不报错、"任务向量方向相反"等说法缺乏直接证据支持。本次修正：

1. **真实观测**：改用 `/mnt/petrelfs/muyao/jiayusun/moe-demo-clean/`（训练 SFT-20k base 所用的同一批 SFT 录制数据）。每个任务从各自的 200 条真实 episode 中取 4 条（对应 batch_size=4），在每条 episode 40% 进度处采样真实头部相机图像、双腕相机图像、真实 qpos，语言指令为该 episode 的真实 resolved 指令（5 个任务各自不同，不再是统一的 "beat block hammer"）。
2. **拆分 self-trajectory / same-trajectory 两种口径**：
   - *self-trajectory*（闭环）：每个模型沿自己的 Euler 轨迹积分，x_t 由该模型自身速度场决定——同时包含参数变化和轨迹分叉两种效应（旧版本的方法）。
   - *same-trajectory*（开环）：所有模型在 SFT-20k base 自己生成的同一组 x_t 序列上探测，不做自身积分——隔离参数效应，排除"两个模型走了不同轨迹"这一混杂因素。
3. **merged 模型分别在 A、B 各自真实观测上评测**，而不是用单一观测代表两个任务。
4. **直接计算任务向量余弦相似度**：τ = θ_PPO − θ_SFT，限定在 action expert（`gemma_expert.*`，201 个参数张量）子空间，为"方向互补/相反"提供实测证据而非仅凭 HiddenSim 大小推测。
5. **修复 `load_weights` 可靠性**：文件不存在时直接抛异常（而非静默返回 False、导致复用上一次残留权重）；`load_state_dict` 的 missing/unexpected keys 严格检查，只放行两种已知良性情况（`value_head` 在 SFT base 中缺失是预期的；`embed_tokens` 因 PaliGemma 的 tied embedding 在 HF 导出时被去重，`lm_head.weight` 加载后会原地同步到共享张量）。
6. **t 的含义澄清**：t 是 flow-matching 去噪的时间步（模型从纯噪声 t=1.0 积分到 t=0.1 的 10 个 Euler 步），本文档**不再**将 t 映射为"任务执行的物理阶段"（如"中途规划""精细执行"），因为没有独立证据支持这种对应关系。

局限性见第 8 节。

---

## 1. 背景：为什么需要 Euler 轨迹方法

### 旧方法的缺陷（固定 x_t，来自 `analyze_hidden_states_moe.py`）

```python
# 旧方法（错误）：对所有 t，x_t 固定为同一份纯噪声
x_t = torch.randn(...)  # 在循环外初始化，固定不变
for t_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
    run_denoising_step(model, ..., x_t, t_val, device)
```

Flow Matching 模型在训练时，对时间步 t 的输入有明确的含义：t 接近 1.0 时 x_t 应接近纯噪声，t 接近 0.0 时应接近干净动作。若对所有 t 都输入纯噪声，则在 t 较小时模型接收到的是严重偏离训练分布的输入，隐藏状态差异反映的是"分布外输入"造成的混乱，而非真实的去噪动态。

### Euler ODE 轨迹采样（本文档方法的基础，仍然正确）

10 步 Euler 积分，从 x₁（纯噪声）→ x₀（干净动作）：

```python
x = x_1
dt = -0.1
for i in range(10):
    t_val = 1.0 + i * dt   # 1.0, 0.9, ..., 0.1
    suffix_out = model.get_suffix_out(state, pfx_pads, pkv, x, t_tensor)
    v_t = model.action_out_proj(suffix_out)
    x = x + dt * v_t
```

这部分实现与模型的 10 步 flow-ODE 推理一致，数值也与 SLURM 日志核对无误——这是唯一在旧版本审阅中被确认"实现正确"的部分。**但旧版本对"训练分布"的操作化定义（观测）是错的**，本次予以修正。

### HiddenSim 定义（不变）

对每个 (layer, t, batch, token)：

```
dist = ||h_A - h_B||₂ / (||h_A||₂ + ||h_B||₂ + ε)
```

在 batch 和 token 维度上取均值，得到 shape = (18 layers, 10 t-steps) 的热力图。

---

## 2. 单任务 PPO vs SFT-20k base（真实观测）

基础模型：`pi05_aloha_moe_demo_clean` checkpoint step 20000（"SFT-20k base"）。观测来自各任务真实 SFT 录制 episode（见第 0 节）。

| 任务 | 简称 | PPO 步数 | self mean | self max | same mean | same max | 热点 (self / same) |
|------|------|---------|-----------|----------|-----------|----------|---------------------|
| beat_block_hammer | bbh | 300 | 0.0459 | **0.1765** | 0.0451 | 0.1766 | layer 17, t=0.2（两者一致） |
| move_can_pot | mcp | 400 | 0.0564 | **0.2436** | 0.0550 | 0.2431 | layer 17, t=0.6（两者一致） |
| place_a2b_left | pal | 300 | 0.0405 | **0.1336** | 0.0394 | 0.1344 | layer 17, t=0.4（两者一致） |
| place_a2b_right | par | **50** | 0.0132 | 0.0299 | 0.0128 | 0.0298 | layer 17, t=0.6（两者一致） |
| place_phone_stand | pps | 300 | 0.0220 | 0.0599 | 0.0212 | 0.0607 | layer 17, t=0.4（两者一致） |

**par 只训练了 50 步**（其余任务 300-400 步），是本组里唯一的欠训练 checkpoint，其偏移量最小很可能主要反映"训练量少"，而非任务本身对 action expert 改动天然保守——不应与其它任务并列做"策略保守程度"的直接比较。

**self ≈ same-trajectory**：本组所有 checkpoint 上，两种口径的 max 相差都在 0.0001-0.0008 之间，热点 (layer, t) 完全一致。这说明在当前训练步数（50-400 步 PPO）下，**推理轨迹分叉对 HiddenSim 的贡献可以忽略**，观测到的差异几乎全部来自参数变化本身。这是一个实证结论，不能外推到训练步数更多、参数改动更大的场景——因此仍建议默认同时报告两种口径（见第 8 节局限性）。

**热点层始终是 layer 17**（action expert 的最后一层），5 个任务一致，比旧版本（分散在 layer 15-17）更集中——真实观测下的信号比合成观测更干净。

### 与合成观测结果的对比

| 任务 | 旧版本（合成观测+错误指令）max @t | 本次（真实观测+正确指令，self）max @t | 变化幅度 |
|------|-----------------------------------|----------------------------------------|----------|
| bbh | 0.1796 @ 0.2 | 0.1765 @ 0.2 | −1.7%，热点不变 |
| pps | 0.0712 @ 0.4 | 0.0599 @ 0.4 | −15.9%，热点不变 |
| pal | 0.1719 @ 0.4 | 0.1336 @ 0.4 | −22.3%，热点不变 |
| par | 0.0759 @ 0.2 | 0.0299 @ 0.6 | **−60.6%，热点也变了** |
| mcp | 0.2504 @ 0.6 | 0.2436 @ 0.6 | −2.7%，热点不变 |

4/5 个任务（bbh、pps、pal、mcp）的量级排序和热点 t 在合成→真实观测切换后基本保持稳定；**par 变化最大**——与它是唯一的欠训练（50 步）checkpoint 吻合，说明 par 的旧结果对输入分布最敏感、最不可信。这为"哪些结论稳健、哪些结论脆弱"提供了直接证据，而不是笼统地说"旧结果无效"或"旧结果照样成立"。

---

## 3. 任务向量余弦相似度（action expert 子空间）

τ = θ_PPO − θ_SFT，限定在 `paligemma_with_expert.gemma_expert.*`（201 个参数张量，与 HiddenStateCollector 探测的层完全对应）。

| 对 | cos(τ_A, τ_B) |
|------|----------------|
| bbh+pps | +0.0042 |
| bbh+pal | +0.0052 |
| bbh+par | +0.0198 |
| bbh+mcp | +0.0080 |
| pps+pal | +0.0271 |
| pps+par | +0.0170 |
| pps+mcp | +0.0082 |
| pal+par | +0.0172 |
| pal+mcp | +0.0198 |
| par+mcp | +0.0052 |

**全部为小正值（+0.004 ~ +0.027），没有一对是负值**，即在 action expert 参数空间里，这 5 个任务的 τ 彼此接近正交、略微同向，不存在"方向相反"的情况。这一节直接为第 5、6 节的机制讨论提供依据。

---

## 4. 配对合并分析（Task Arithmetic α=0.7）：merged 分别在 A、B 各自真实观测上评测

合并公式：θ_merged = θ_SFT + 0.7 × (τ_A + τ_B)。下表 `merged@A-obs` / `merged@B-obs` 分别是把 merged 模型放到任务 A / 任务 B 各自真实观测下评测得到的 self-trajectory HiddenSim max（same-trajectory 与之相差普遍 <0.001，趋势一致，不单独列出，完整数值见 `heatmap_<ta>_<tb>.npz`）。AvgΔ性能取自 `2026-7-6-pairwise-merge-hiddensim.md` 的 eval 结果，两任务 Δ 的平均值。

| 对 | A solo max | merged@A-obs | B solo max | merged@B-obs | cos(τ_A,τ_B) | AvgΔ性能 |
|------|-----------|--------------|-----------|--------------|---------------|-----------|
| bbh+pps | 0.1765 | 0.1383 | 0.0599 | 0.1398 | +0.0042 | −8.25pp |
| bbh+pal | 0.1765 | 0.1949 | 0.1336 | 0.1975 | +0.0052 | −17.05pp |
| bbh+par | 0.1765 | 0.1414 | 0.0299 | 0.1420 | +0.0198 | −3.75pp |
| bbh+mcp | 0.1765 | 0.2314 | 0.2436 | 0.2326 | +0.0080 | −21.05pp |
| pps+pal | 0.0599 | 0.1284 | 0.1336 | 0.1280 | +0.0271 | −0.95pp |
| **pps+par** | 0.0599 | **0.0449** | 0.0299 | **0.0436** | +0.0170 | **+6.70pp** |
| pps+mcp | 0.0599 | 0.1521 | 0.2436 | 0.1516 | +0.0082 | −7.95pp |
| pal+par | 0.1336 | 0.0994 | 0.0299 | 0.0999 | +0.0172 | +5.75pp |
| pal+mcp | 0.1336 | 0.1899 | 0.2436 | 0.1904 | +0.0198 | −6.15pp |
| par+mcp | 0.0299 | 0.1823 | 0.2436 | 0.1826 | +0.0052 | 0.00pp |

### 关键案例分析

#### pps+par：最佳合并对（AvgΔ +6.70pp），merged 在两个观测上都是 10 对里最小的

merged@pps-obs=0.0449、merged@par-obs=0.0436，是全表最小的两个 merged 值。**但这不能归因于"τ_pps 与 τ_par 方向互补/相反"**——它们的余弦相似度 +0.0170 与另外 6 对处于同一量级，甚至比 bbh+pps（+0.0042）、par+mcp（+0.0052）更"同向"而非更"相反"。更直接的解释是：pps（0.0599）和 par（0.0299）恰好是 5 个任务里 solo HiddenSim **最小的两个**（par 还是唯一欠训练的），两个小改动叠加，merged 自然还接近 base。旧版本"pps 和 par 任务向量方向部分相反、合并后相互抵消"的表述缺乏依据，予以撤回。

#### bbh+mcp：合并后偏离 base 幅度最大（AvgΔ −21.05pp）

bbh（0.1765）和 mcp（0.2436）恰好是 solo HiddenSim 最大的两个任务，merged 在两个观测上都在 0.231-0.233，接近 mcp 自身的量级——mcp 的 τ 范数占主导，不论用哪个任务的真实观测去探测，merged 模型都显著偏离 base。

#### par+mcp：反例，提醒 HiddenSim 不是性能变化的充分统计量

merged HiddenSim（0.182-0.183）在 10 对里排第 7 大，但 AvgΔ性能≈0.00pp——好于多个 merged 数值更小的对（如 bbh+pps 的 −8.25pp）。说明 HiddenSim 大小只是合并对参数改动幅度的一个粗糙代理，不能单独用来预测性能变化的方向和幅度。

---

## 5. 对合并成功率的预测能力

用 avg(merged@A-obs, merged@B-obs) 与 AvgΔ性能 做等级相关（Spearman），10 对全量数据：

| 排名（merged 均值从小到大） | 对 | avg merged max | AvgΔ性能 |
|---|------|-----------------|-----------|
| 1 | pps+par | 0.0443 | +6.70pp |
| 2 | pal+par | 0.0997 | +5.75pp |
| 3 | pps+pal | 0.1282 | −0.95pp |
| 4 | bbh+pps | 0.1391 | −8.25pp |
| 5 | bbh+par | 0.1417 | −3.75pp |
| 6 | pps+mcp | 0.1519 | −7.95pp |
| 7 | par+mcp | 0.1825 | 0.00pp |
| 8 | pal+mcp | 0.1902 | −6.15pp |
| 9 | bbh+pal | 0.1962 | −17.05pp |
| 10 | bbh+mcp | 0.2320 | −21.05pp |

avg merged HiddenSim 与 AvgΔ性能 的 Spearman 相关系数 ρ ≈ **−0.77**（中等偏强的负相关：HiddenSim 越大、AvgΔ性能越负；等价地说，HiddenSim 越小，平均性能变化倾向越正）。两端吻合得很好——最小(pps+par)对应最优 AvgΔ，最大(bbh+mcp)对应最差 AvgΔ——但**中间样本不严格单调**，par+mcp 是明确的反例（排名第 7 大但 AvgΔ≈0）。

旧版本"merged HiddenSim 越小、合并点越接近 SFT base、合并性能越优"被称为"核心规律"并声称"验证了机制"，证据强度被夸大了。更准确的表述是：**HiddenSim 大小与合并后的性能变化存在中等强度的等级相关，在两端（最好/最差对）表现稳健，但不是可靠的逐对预测器**（par+mcp 是反例）。

---

## 6. 对合并理论的启示（修正版）

1. **任务向量的范数（有较强证据支持）**：用 solo HiddenSim 峰值作为 τ 范数的代理指标，mcp 的范数明显大于其它任务；凡是与 mcp 配对，merged 在两个观测上都显著偏离 base、贴近 mcp 自身量级。pps+par（两个 solo 峰值最小的任务）合并后最接近 base。这是本次数据里唯一有直接、一致证据支持的因素。
2. **任务向量的方向（证据不支持）**：10 对余弦相似度全为 +0.004~+0.027 的小正值，无一为负。不存在"方向互补/相反"的情况，pps+par 的余弦也没有特殊之处。旧版本"pps 和 par 方向互补→合并后相互抵消"的表述已被本次直接测量推翻。
3. **热点时间步的重叠**：在有了范数和方向的直接证据后，这个因素的解释力显得次要——bbh(t=0.2) 和 mcp(t=0.6) 热点不重叠，但合并后仍严重偏离 base，主要原因还是 mcp 范数过大，而非"重叠与否"。不再单独作为独立的解释变量。
4. **HiddenSim 不是性能的充分统计量**：par+mcp 的反例说明，表征层面的漂移只是"模型改动幅度"的粗糙代理，实际任务性能还受具体任务组合的其它因素影响，本次实验未覆盖。

### 对 rf20sft 融合实验的解释（推测，未在本次真实观测重跑范围内验证）

旧版本曾用本文的（当时有缺陷的）结果去解释另一个实验（rf20sft）里 CA/adj 融合成功的原因。这属于跨实验的推测性外推，本次真实观测重跑没有覆盖 rf20sft 的 checkpoint，因此该推测既没有被加强也没有被证伪，仅供参考，不应视为已验证的结论。

---

## 7. 与更早分析（`analyze_hidden_states_moe.py`）的对比

| 分析方法 | bbh max HiddenSim | 热点 t |
|----------|-------------------|--------|
| 最早方法（固定 x_t，OOD 输入） | ~0.05 | t=0.1–0.3（输入分布外，不可信） |
| Euler 轨迹 + 合成观测（本文档旧版本） | 0.1796 | t=0.2 |
| Euler 轨迹 + 真实观测（本次，self-trajectory） | **0.1765** | t=0.2 |

最早方法在小 t 时输入的是纯噪声（而非接近干净动作的 x_t），模型在分布外输入上的行为混乱，掩盖了真实的表征差异；这一结论依然成立。Euler 轨迹方法本身（x_t 正确演化）在合成观测和真实观测下给出的 bbh 数值高度接近（0.1796 vs 0.1765），说明该任务的结果对观测真实性不敏感——但如第 2 节所示，**这不能推广到所有任务**（par 的变化幅度达 60%）。

---

## 8. 局限性

1. **观测来源**：真实图像/qpos/指令来自 `moe-demo-clean` 的 SFT 录制 episode（训练 SFT-20k base 的同一批数据），是训练分布内的真实内容，但**不是 PPO on-policy rollout 本身产生的帧**，不包含 PPO 训练/推理时的 domain randomization 或 on-policy 状态分布偏移。
2. **单帧采样**：每条 episode 只固定取 40%（`REAL_OBS_FRACTION=0.4`）进度的一帧，batch_size=4（每任务 4 条不同 episode），没有做跨时间点或更大 batch 的敏感性检验。若某任务在 episode 早/晚期的表征漂移模式不同，本次结果不会体现。
3. **任务向量余弦相似度仅限 action expert 子空间**（201 个参数张量），不包含 vision encoder / language backbone 的参数变化；若这些部分的方向关系不同，结论可能不同。
4. **self ≈ same-trajectory 的结论仅在本组 checkpoint（50-400 PPO 步）下成立**，不代表更大训练量、更大参数改动下轨迹分叉依然可忽略。
5. **t 表示 flow-matching 去噪时间步**，不代表机器人任务执行的物理阶段，本文档不再做"计划阶段/精细执行阶段"一类的映射。
6. **HiddenSim 大小与合并性能的等级相关（|ρ|≈0.77）建立在 10 个样本上**，样本量小，par+mcp 已是明确反例，不宜过度泛化。

---

## 9. 输出文件索引

```
results/euler_trajectory_heatmaps_v2/
├── 单任务（PPO vs SFT-20k base，真实观测）：5 × (png + npz)
│   ├── euler_hs_bbh_vs_base.png/.npz     self max=0.1765 same max=0.1766 @layer17,t=0.2
│   ├── euler_hs_pps_vs_base.png/.npz     self max=0.0599 same max=0.0607 @layer17,t=0.4
│   ├── euler_hs_pal_vs_base.png/.npz     self max=0.1336 same max=0.1344 @layer17,t=0.4
│   ├── euler_hs_par_vs_base.png/.npz     self max=0.0299 same max=0.0298 @layer17,t=0.6（仅 50 PPO 步）
│   └── euler_hs_mcp_vs_base.png/.npz     self max=0.2436 same max=0.2431 @layer17,t=0.6
│
├── 配对合并（2行[self/same] x 4列[A@A-obs, merged@A-obs, B@B-obs, merged@B-obs]）：10 × (png + npz)
│   ├── heatmap_bbh_pps.png/.npz    merged@bbh-obs=0.1383  merged@pps-obs=0.1398
│   ├── heatmap_bbh_pal.png/.npz    merged@bbh-obs=0.1949  merged@pal-obs=0.1975
│   ├── heatmap_bbh_par.png/.npz    merged@bbh-obs=0.1414  merged@par-obs=0.1420
│   ├── heatmap_bbh_mcp.png/.npz    merged@bbh-obs=0.2314  merged@mcp-obs=0.2326
│   ├── heatmap_pps_pal.png/.npz    merged@pps-obs=0.1284  merged@pal-obs=0.1280
│   ├── heatmap_pps_par.png/.npz    merged@pps-obs=0.0449  merged@par-obs=0.0436 ← 全表最小
│   ├── heatmap_pps_mcp.png/.npz    merged@pps-obs=0.1521  merged@mcp-obs=0.1516
│   ├── heatmap_pal_par.png/.npz    merged@pal-obs=0.0994  merged@par-obs=0.0999
│   ├── heatmap_pal_mcp.png/.npz    merged@pal-obs=0.1899  merged@mcp-obs=0.1904
│   └── heatmap_par_mcp.png/.npz    merged@par-obs=0.1823  merged@mcp-obs=0.1826
│
└── task_vector_cosine.json         10 对 action-expert 任务向量余弦相似度（第 3 节）
```

**脚本**：`scripts/plot_euler_trajectory_heatmaps.py`
**SLURM**：`examples/embodiment/slurm/plot_euler_trajectory_heatmaps.sbatch`
**SLURM log**：`slurm_logs/plot_euler_trajectory_heatmaps-10204865.out`
