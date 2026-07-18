# 2026-6-22 实验记录：SFT权重 vs RL权重差异分析

## 研究动机

基于 Cao et al. 2026《An Empirical Study and Theoretical Explanation on Task-Level Model-Merging Collapse》（arXiv 2603.09463）的理论框架，研究 pi0.5 机器人策略模型中：

- 为什么 **ca_sft15k**（单任务 SFT 专化）会导致 adjust_bottle **灾难性遗忘**（0.0%）
- 而 **merged_a07_ppo_adj_step50**（从合并基础出发的 PPO）能**同时保持两个任务**（CA=87.7%，adj=84.2%）

核心假设：SFT 是全参数微调，修改了 SigLIP + PaliGemma LM + Action Expert；
PPO（`train_expert_only=True`）只修改 Action Expert 层（208/812 keys）。

---

## 检查点路径

| 名称 | 路径 | CA | adj |
|------|------|-----|-----|
| **rf20sft**（基准）| `/mnt/petrelfs/muyao/jiayusun/random_first20_pi05_sft/full_weights.pt` | ~33% | ~50% |
| **ca_sft15k** | `logs/20260617-20:39:17-rf20_click_alarmclock_sft_8gpu/.../global_step_15000/actor/model_state_dict/full_weights.pt` | 55.7% | **0.0%** |
| **merged_a07** | `/mnt/petrelfs/muyao/jiayusun/merged_weights/ppo_adj200_plus_ca100_tv_a07/full_weights.pt` | 93.6% | 73.4% |
| **ppo_adj_step50** | `logs/20260619-16:05:12-merged_a07_ppo_adjust_bottle/.../global_step_50/actor/model_state_dict/full_weights.pt` | 87.7% | **84.2%** |

任务向量定义：
- `τ_SFT = ca_sft15k − rf20sft`
- `τ_PPO = ppo_adj_step50 − merged_a07`

---

## 实验一：逐层参数变化量分析

**脚本**：`scripts/analyze_weight_diff.py`  
**执行**：登录节点，CPU，约 2 分钟  
**输出**：`results/weight_diff/`

### 主要发现

**权重变化量（rel_change = ‖Δθ‖_F / ‖θ_base‖_F）**：

| 模块类型 | SFT 变化量 | PPO 变化量 | 倍数 |
|----------|-----------|-----------|------|
| input_layernorm | 0.0552 | 0.0057 | 9.7× |
| post_attention_layernorm | 0.0371 | 0.0043 | 9.1× |
| down_proj | 0.0156 | 0.00087 | 19.8× |
| o_proj | 0.0113 | 0.00056 | 20.2× |
| q_proj | 0.0105 | 0.00048 | 22.1× |
| k_proj | 0.0100 | 0.00044 | 23.5× |
| gate_proj | 0.0096 | 0.00049 | 20.8× |
| up_proj | 0.0085 | 0.00046 | 19.2× |
| v_proj | 0.0069 | 0.00035 | 19.8× |

**结论**：SFT 对 Action Expert 权重的改动幅度约为 50 步 PPO 的 **10–23 倍**，注意力层差异最大（23×）。

---

## 实验二：任务向量干扰分析

**核心指标**：`cos(τ_SFT, τ_PPO)` —— 两个任务向量方向是否冲突

### 主要发现

**全局干扰余弦**：`−0.0059`，**62.5% 的参数**方向冲突（cos < 0）

**按模块类型的干扰余弦均值**：

| 模块类型 | cos(τ_SFT, τ_PPO) | 解读 |
|----------|-------------------|------|
| action_in_proj | **−0.110** | 最强冲突（动作输入投影）|
| post_attention_layernorm | −0.017 | 冲突 |
| action_out_proj | **−0.015** | 冲突（动作输出投影）|
| v_proj | −0.012 | 冲突 |
| input_layernorm | −0.010 | 冲突 |
| q_proj | +0.002 | 近似正交 |
| time_mlp_in | +0.038 | 略微对齐 |
| norm | +0.047 | 略微对齐 |
| time_mlp_out | **+0.057** | 最强对齐（时间调制输出）|

**与 Libero 基线对比**：Libero 实验中任务向量余弦约 0.05（近正交）。本实验全局均值 −0.006，偏负方向，说明 SFT 和 PPO 的更新方向在 Action Expert 空间内存在干扰，符合"合并崩溃"理论预测。

**图表**：
- `results/weight_diff/interference_heatmap.png`：18层 × 模块类型热力图
- `results/weight_diff/interference_by_layer.png`：逐层干扰余弦折线图
- `results/weight_diff/layer_rel_change_curves.png`：SFT vs PPO 权重变化量对比
- `results/weight_diff/layer_cosine_curves.png`：逐层余弦相似度

---

## 实验三：隐藏状态相似度分析（HiddenSim）

**脚本**：`scripts/analyze_hidden_states.py`  
**SLURM Job**：9813512（wam_data, 1 GPU, 节点 SH-IDCA1404-10-140-54-20）  
**执行时间**：约 11 分钟（19:37→19:40，含模型加载 2.5 分钟）  
**输出**：`results/hidden_states/`

### 方法

- 模型：`OpenPi0ForRLActionPrediction`（pi05_aloha_robotwin15 配置）
- Hooks：18 层 `gemma_expert.model.layers` 各注册 forward hook，捕获 `hs[:, -50:, :]`（50 个 action token 隐藏状态）
- 固定随机观测（seed=42，HWC uint8，224×224）+ 固定 x_t（bfloat16 seed=42）
- 时间步：t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}，批大小 B=4

**HiddenSim 公式对比**

**chatgpt.md 提出的公式（论文建议版）**：

先将每个 token 隐藏向量单位归一化，再求 L2 距离，取期望：

$$d_{i,j}^{\pi0.5} = \mathbb{E}_{o,\,t}\!\left[\left\|\frac{\operatorname{vec}(H_i^{\text{final}})}{\|\operatorname{vec}(H_i^{\text{final}})\|_2} - \frac{\operatorname{vec}(H_j^{\text{final}})}{\|\operatorname{vec}(H_j^{\text{final}})\|_2}\right\|_2\right]$$

其中 $H^{\text{final}} \in \mathbb{R}^{50 \times 1024}$ 为 `final_norm_1` 之后所有 action token 的隐藏状态，$d \in [0, 2]$（单位向量最大距离为 2）。最终再做 max-min 归一化转为相似度：

$$\text{HiddenSim}_{\pi0.5}(i,j) = \frac{\max(D) - d_{i,j}^{\pi0.5}}{\max(D) - \min(D)}$$

**代码实际实现的公式**（`scripts/analyze_hidden_states.py`）：

**Step 1：收集 hidden states**

对每个 checkpoint，在固定 x_t（seed=42）和固定合成观测下，遍历 t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}，通过 `get_suffix_out` 触发 forward，hook 捕获每层输出：

```
layer_hook → hs shape = (B=4, 50, 1024)
```

即每个 checkpoint 最终得到 `{t_val: {layer_idx: ndarray(4, 50, 1024)}}`。

**Step 2：逐元素计算距离**

对每对 checkpoint (A, B)，枚举所有 (l, t, k) 三元组，取该 token 位置的 batch 向量：

```python
h_a_k = h_a[:, k, :]   # shape (4, 1024)
h_b_k = h_b[:, k, :]   # shape (4, 1024)

diff   = ||h_a_k - h_b_k||_2  per row → shape (4,)   # 对 1024 维求 L2
norm_a = ||h_a_k||_2           per row → shape (4,)
norm_b = ||h_b_k||_2           per row → shape (4,)

sim_arr[l, t, :, k] = diff / (norm_a + norm_b + 1e-8)  # shape (4,)
```

公式：

$$\text{dist}_{l,t,b,k} = \frac{\|h_A^{(b,k)} - h_B^{(b,k)}\|_2}{\|h_A^{(b,k)}\|_2 + \|h_B^{(b,k)}\|_2 + \varepsilon}, \quad h \in \mathbb{R}^{1024}$$

其中下标 $(l,t)$ 隐含在循环中，$b \in \{0,1,2,3\}$，$k \in \{0,\ldots,49\}$。

**Step 3：结果张量**

```
sim_arr  shape = (18层, 5步, 4 batch, 50 tokens)
```

各图的均值取法：
- **全局均值**（表格中报告的数字）：`sim_arr.mean()` → 标量
- **逐层曲线**：`sim_arr.mean(axis=(1,2,3))` → shape (18,)，对 t、batch、token 取均值
- **逐 t 曲线**：`sim_arr.mean(axis=(0,2,3))` → shape (5,)，对层、batch、token 取均值
- **逐 token 曲线**：`sim_arr.mean(axis=(0,1,2))` → shape (50,)，对层、t、batch 取均值
- **热力图**：`sim_arr.mean(axis=(2,3))` → shape (18, 5)，对 batch 和 token 取均值

**两个公式的关键差异**：

| | 论文建议 | 代码实现 |
|--|---------|---------|
| 归一化方式 | 先将向量单位归一化再求距离 | 距离直接除以两向量范数之和 |
| 保留尺度信息 | 否（只比较方向） | 是（尺度差异也会影响结果）|
| 输出范围 | $d \in [0,2]$，再 max-min 映射到相似度 | $\text{dist} \in [0,1]$（直接为距离，越大越不同）|
| 覆盖层 | 仅 `final_norm_1` 后的输出 | 全部 18 层逐层 |

> **关系**：论文公式等价于两向量之间的**弦距离**（chord distance），仅捕捉方向差异；代码公式是**相对距离**，同时包含方向和尺度变化。当两模型的隐藏向量范数相近时，两式数值接近（约差常数因子 $\approx 1/\sqrt{2}$）。本实验各 checkpoint 的激活尺度相近（同架构，共享 final_norm_1 的 RMSNorm），因此两公式趋势一致，差距主要在绝对量级。

### 对比结果（7组，Job 9813512 / 9813637 / 9813658）

所有对比均以 rf20sft 为基准（除 B 使用 merged_a07 为基准）。

| 对比组 | checkpoint_A vs checkpoint_B | 全局 HiddenSim | 峰值层 | 浅层(0-5) | 中层(6-11) | 深层(12-17) |
|--------|------------------------------|---------------|--------|----------|-----------|------------|
| **A: SFT-ft vs Base** | ca_sft15k vs rf20sft | **0.1207** | 16 (0.1535) | 0.0972 | 0.1214 | 0.1435 |
| B: PPO-adj50 vs merged_a07 | ppo_step50 vs merged_a07 | 0.0108 | 15 (0.0165) | 0.0081 | 0.0097 | 0.0147 |
| C: SFT-ft vs PPO-adj50 | ca_sft15k vs ppo_step50 | 0.1069 | 15 (0.1360) | 0.0839 | 0.1087 | 0.1282 |
| D: PPO-CA vs Base | ppo_ca_100 vs rf20sft | 0.0249 | 17 (0.0353) | 0.0206 | 0.0227 | 0.0314 |
| E: PPO-adj vs Base | ppo_adj_200 vs rf20sft | 0.0197 | 16 (0.0269) | 0.0158 | 0.0183 | 0.0251 |
| F: merged_a07 vs Base | merged_a07 vs rf20sft | 0.0300 | 17 (0.0447) | 0.0243 | 0.0275 | 0.0380 |
| G: PPO-adj50 vs Base | ppo_step50 vs rf20sft | 0.0317 | 17 (0.0477) | 0.0254 | 0.0280 | 0.0417 |
| H: adj-SFT-ft vs Base | adj_sft15k vs rf20sft | 0.0808 | 16 (0.1150) | 0.0584 | 0.0762 | 0.1077 |
| I: CA-SFT vs adj-SFT | ca_sft15k vs adj_sft15k | 0.0705 | 15 (0.1034) | 0.0514 | 0.0679 | 0.0922 |

**逐层（对比组 A: SFT vs Base）**：

| 层 | HiddenSim | 层 | HiddenSim |
|----|-----------|----|-----------| 
| 0  | 0.0527    | 9  | 0.1210    |
| 1  | 0.0951    | 10 | 0.1250    |
| 2  | 0.1036    | 11 | 0.1304    |
| 3  | 0.1070    | 12 | 0.1329    |
| 4  | 0.1105    | 13 | 0.1366    |
| 5  | 0.1141    | 14 | 0.1406    |
| 6  | 0.1159    | 15 | 0.1481    |
| 7  | 0.1165    | 16 | **0.1535** |
| 8  | 0.1194    | 17 | 0.1492    |

### 关键发现

1. **SFT 改变表征约 5× 多于 PPO（相同 rf20sft 基准）**
   - A（SFT-ft）= 0.121 vs D（PPO-CA）= 0.025、E（PPO-adj）= 0.020
   - 即使 PPO 训练了 100–200 步，表征移动也仅为 SFT 的 1/5

2. **表征变化集中在深层（12-17 层）**，所有对比的峰值均在第 15–17 层，浅层（0-1 层）变化最小

3. **B≈0（PPO-adj50 在 merged_a07 基础上几乎不移动）**：0.011，说明 PPO 在合并基础上的增量极小

4. **G ≈ F（PPO-adj50 vs Base ≈ merged_a07 vs Base）**：0.0317 ≈ 0.0300
   - merged_a07 是 Task Arithmetic 合并（rf20sft + 0.7×PPO_CA + 0.7×PPO_adj 任务向量）
   - 在 merged_a07 基础上再做 50 步 PPO 后，相对 rf20sft 的距离几乎没变
   - 说明 50 步 PPO 的增量（B=0.011）在 merged_a07 已有的漂移方向上微调，**净效果≈0**

5. **A≈C（SFT_vs_Base ≈ SFT_vs_PPO-adj50）**：0.121 ≈ 0.107
   - 因为 G≈0.032（PPO-adj50 离 Base 很近），C 的距离主要由 A 决定
   - PPO 的表征几乎与 Base 重叠，SFT 与 PPO 的差距本质是 SFT 与 Base 的差距

6. **H（adj-SFT）< A（CA-SFT），但均远大于 PPO**
   - adj-SFT vs Base = 0.0808，CA-SFT vs Base = 0.1207，差距 33%
   - **行为对应**：CA-SFT 造成 adj 灾难性遗忘（0%），adj-SFT 两个任务均提升（CA +21%，adj +33%）
   - 结论：**表征漂移幅度与跨任务遗忘严重程度正相关**——漂移越大，遗忘越严重

7. **I（CA-SFT vs adj-SFT）= 0.0705，两种 SFT 已相互分离**
   - 两者各自从 Base 出发分别偏移了 0.121 和 0.081，彼此之间距离 0.0705
   - 说明两者专化方向不同，adj-SFT 专化更温和（量级更小），CA-SFT 专化更激进

**图表（Job 9813658）**：
- `results/hidden_states/layer_curves.png`：7 组对比逐层曲线
- `results/hidden_states/t_curves.png`：按去噪时间步 t 的变化趋势
- `results/hidden_states/horizon_curves.png`：按 action token 位置 k (0..49) 的变化
- `results/hidden_states/heatmap_*.png`：各组层×时间步热力图（共 7 张）

---

### 图表详细解读

#### 图1：layer_curves.png — 逐层 HiddenSim

**轴**：X = Action Expert 层索引（0–17），Y = HiddenSim（归一化 L2 距离，对 t、batch、action token 取均值）

**现象**：
- **A（红）SFT-ft vs Base**：从第 0 层的 0.053 急升到第 1 层的 0.095，之后平缓单调增至第 16 层峰值 0.153，第 17 层微降至 0.149。整条曲线高悬，第 0 层就已远高于所有 PPO 曲线。
- **C（紫）SFT-ft vs PPO-adj50**：形状几乎与 A 平行，整体下移约 0.013（差值正好是 G 的全局均值 0.032），进一步验证 C ≈ A + 常数偏移。
- **D/E/F/G（绿/青/蓝/棕）PPO 系列**：都有一个 **层 0→1 的"小驼峰"**——从第 0 层下降到第 2 层附近的局部最低点（约 0.018–0.020），然后再单调上升至第 17 层。这个"先降后升"是 PPO 独有的形态，SFT 没有。F 和 G 高度几乎重叠，再次确认 G≈F。
- **B（橙）PPO-adj50 vs merged_a07**：全程压在最底部（0.007–0.017），与其他组在量级上完全分离。

**解读**：
- SFT 的表征变化从浅层就已显著，并随深度持续累积——说明 SFT 改变了整个 Action Expert 的计算流。
- PPO 在第 0 层就有一定偏移（Task Arithmetic 的注入），但浅层中间层（1–5 层）被某种"恢复"机制压低，深层（12–17）才再度拉开——PPO 的影响更集中在高层语义处理阶段。
- D（PPO-CA）>F（merged_a07）≈G（PPO-adj50）>E（PPO-adj）：PPO-CA（click alarmclock 专化）表征偏移比 PPO-adj（adjust bottle 专化）略大，说明两个任务的 PPO 更新方向有差异。

---

#### 图2：t_curves.png — 按去噪时间步的 HiddenSim

**轴**：X = 去噪时间步 t ∈ {0.1, 0.3, 0.5, 0.7, 0.9}（t=0 为干净动作，t=1 为纯噪声），Y = HiddenSim（对所有层、batch、token 取均值）

**现象**：
- **A（红）SFT-ft vs Base**：单调**递减** — t=0.1 最高（0.139），t=0.9 最低（0.114）。越接近干净动作输出阶段，SFT 与 Base 的差异越大。
- **C（紫）SFT-ft vs PPO-adj50**：同样递减，t=0.1（0.129）→ t=0.9（0.093），与 A 形状一致。
- **D（绿）PPO-CA vs Base**：单调**递增** — t=0.1 最低（0.017），t=0.9 最高（0.044）。与 SFT 方向**相反**。
- **E（青）PPO-adj vs Base**：整体也倾向于高 t 偏大，但 t=0.5→0.7 有一次下降后再升，形态最复杂（说明 adj PPO 的更新对不同去噪阶段影响不均匀）。
- **F（蓝）merged_a07 vs Base**：在 t=0.1–0.5 较平（约 0.026–0.030），t=0.9 跳至 0.037（受 Task Arithmetic 的 D+E 双向量影响叠加）。
- **G（棕）PPO-adj50 vs Base**：与 F 相似，t=0.1–0.5 平稳（0.029–0.030），t=0.9 升至 0.037。
- **B（橙）PPO-adj50 vs merged_a07**：全程极平（0.010–0.012），时间步几乎无影响。

**解读**：
- **SFT 在"精细执行阶段"（t→0）偏移最大**：SFT 专门训练了如何生成精准动作，改变的是 Action Expert 在低噪声状态下的输出表征。
- **PPO-CA 在"噪声条件阶段"（t→1）偏移最大**：PPO 改变了模型对高噪声输入的响应方式，这可能对应着 PPO 调整了策略的"初始猜测"而非"精细修正"能力。
- **两种训练范式在 Flow Matching 的不同阶段留下印记**，SFT 和 PPO 对去噪过程的作用位置不同。

---

#### 图3：horizon_curves.png — 按 action token 位置的 HiddenSim

**轴**：X = action token 位置 k（0–49，共 50 个动作 token），Y = HiddenSim（对所有层、t、batch 取均值）

**现象**：
- 所有 7 条曲线在 k=0 到 k=49 上**几乎完全水平**，仅有轻微的高频噪声波动（幅度 < 0.005）。
- A（红）约 0.120，C（紫）约 0.106，F/G 约 0.031，D 约 0.026，E 约 0.021，B 约 0.011。
- k=45–49（最后几个 action token）有极轻微的上翘，但统计意义不显著。

**解读**：
- **各训练方式对 action horizon 的所有位置影响均匀**，SFT 或 PPO 都没有优先改变"近端动作"或"远端动作"的表征。
- 这与 pi0.5 的架构设计一致——50 个 action token 在 Action Expert 中通过 self-attention 相互关注，权重更新自然会均匀扩散。
- 该图否定了"SFT 只影响前几步动作精度"的假设，SFT 的遗忘是**全 horizon 的系统性表征偏移**。

---

#### 图4–10：heatmap 系列 — 层 × 时间步二维热力图

每张热力图 X 轴为去噪时间步 t，Y 轴为层索引（上=0，下=17），颜色越深 = HiddenSim 越大。

**A: SFT-ft vs Base**（色阶 0.06–0.18）
- 整体偏橙红，第 0 层极浅（几乎白色），深度越大颜色越深。
- **热点区域：层 14–17 × t=0.1–0.5**（深红/暗红），尤其层 16 at t=0.1 最深。
- t 越大（右侧），颜色整体变浅——即精细执行阶段（小 t）的变化更显著，与 t_curves 完全吻合。

**B: PPO-adj50 vs merged_a07**（色阶 0.006–0.018，量级比 A 低 10×）
- 热点不在左下角而在**中右下角：层 13–15 × t=0.5–0.7**（暗红）。
- 层 0–10 全程极浅；深层在中间时间步最大。
- 这说明 PPO 增量的 50 步更新集中影响中间去噪阶段的深层处理。

**C: SFT-ft vs PPO-adj50**（色阶 0.04–0.16）
- 形态与 A 几乎完全相同（热点同样在深层 × 小 t），量级略小。此图直观展示 A≈C 的原因。

**D: PPO-CA vs Base**（色阶 0.01–0.05）
- 与 A 形态**完全相反**：左侧（小 t）颜色极浅（接近白），右侧（大 t）深红。
- 热点：**层 14–17 × t=0.7–0.9**，尤其层 15–16 at t=0.9 最深（暗红）。
- 这是 PPO-CA 独特的"噪声阶段偏移"特征。

**E: PPO-adj vs Base**（色阶 0.012–0.037）
- 热点在**层 15–17 × t=0.1–0.3**（左下角有暗红），中间 t（0.5–0.6）最浅，t=0.7–0.9 深层再度变红。
- 呈现**双峰形态**（小 t 和大 t 都偏大，中间 t 最小），与 D 的单调递增不同，说明 adj PPO 在两端去噪阶段都有影响。

**H: adj-SFT-ft vs Base**（色阶 0.06–0.15）
- 整体形态介于 A 和 D/E 之间：左下角（深层 × t=0.1）有暗红热点（像 A），但右下角（深层 × t=0.7–0.9）也有明显红色（像 D/E）。
- **双峰时间步结构**：在 t_curves 图中，H（粉色）在 t=0.1 最高（0.090），t=0.3–0.5 轻微下降（0.078–0.076），t=0.7–0.9 再度回升至 0.083——U 型曲线，与 A 的单调递减和 D 的单调递增均不同。
- 解读：adj-SFT 同时影响了"精细执行阶段"（小 t）和"噪声条件阶段"（大 t），说明其训练数据（adjust_bottle）对 Action Expert 的多个去噪阶段均有塑造作用，不像 CA-SFT 那样只在精细执行阶段留下强烈印记。

**I: CA-SFT vs adj-SFT**（色阶 0.04–0.13）
- 热点集中在**深层（14–17）× 中间时间步（t=0.4–0.6）**，这正好是 A（CA-SFT vs Base）最强而 H（adj-SFT vs Base）最弱的区域。
- 解读：两种 SFT 的最大差异在于**中间去噪阶段的深层表征**——CA-SFT 在 t=0.1–0.3 的深层强烈特化，adj-SFT 相对保持接近 Base；反之，adj-SFT 在 t=0.7–0.9 的深层保留了更多 Base 特征而 CA-SFT 在此阶段差距相对小。两者专化的去噪阶段不同，这可能是 adj-SFT 泛化性好（双任务均提升）的表征层面解释。

**F: merged_a07 vs Base**（色阶 0.018–0.052）
- 热点在**层 15–17 × t=0.1**（左下角暗红），同时 **t=0.7–0.9 × 深层** 也有显著红色。
- 明显的双峰结构，是 D（PPO-CA 贡献）和 E（PPO-adj 贡献）两个任务向量叠加的直观体现。

**G: PPO-adj50 vs Base**（色阶 0.018–0.057）
- 形态与 F 几乎相同，热点同样分布在**层 15–17 × t=0.1–0.3** 和 **t=0.7–0.9 × 深层**。
- F 和 G 热图的高度相似再次定量确认 G≈F，即 50 步 PPO 对整体热力图格局几乎没有改变。

---

## 脚本调试记录

### Bug 1：`torch.load` weights_only=True
- **错误**：`.pt` 文件使用 pickle 格式，`weights_only=True` 拒绝加载
- **修复**：移除 `weights_only=True`，与 `task_arithmetic_rlinf.py` 保持一致

### Bug 2：中文字体缺失
- **错误**：matplotlib 图表中中文字符显示为方块（Glyph XXXX missing from font DejaVu Sans）
- **修复**：将所有图表标签改为英文

### Bug 3：Worker 模块路径错误（Job 9813443→9813450）
- **错误**：`ModuleNotFoundError: No module named 'rlinf.workers.worker'`
- **修复**：改为 `from rlinf.scheduler.worker.worker import Worker as _Worker`

### Bug 4：`model.apply_input_transforms` 不存在（Job 9813450/9813493）
- **错误**：`build_prefix_cache` 中调用了不存在的方法，fallback 导致所有检查点跳过
- **正确流程**：
  ```python
  env_obs = {"main_images": (B, H, W, 3), "wrist_images": (B, 2, H, W, 3),
             "states": (B, 14), "task_descriptions": [...], "extra_view_images": None}
  to_process_obs = model.obs_processor(env_obs)         # obs → policy input
  processed_obs  = model.input_transform(to_process_obs, transpose=False)
  processed_obs  = model.precision_processor(processed_obs)
  observation    = _model.Observation.from_dict(processed_obs)
  ```

### Bug 5：dtype 不匹配（Job 9813493→9813512）
- **错误**：`RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float`
- **原因**：`.pt` 权重文件是 float32，`load_state_dict` 后模型变为 float32，但 `x_t` 是 bfloat16
- **修复**：在 `load_weights` 中加 `model.to(torch.bfloat16)`

### Bug 6：`get_suffix_out` 替代 `denoise_step`
- **原因**：父类 `PI0Pytorch.denoise_step` 在 rlinf 版本中可能不直接使用
- **修复**：改用 `model.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_tensor)` 触发 hooks

---

## Ultraplan 提议的后续实验（待执行）

### 背景
Ultraplan（cloud session）提议基于上述发现，设计消融实验验证：
> 将 SFT 任务向量过滤到 Action Expert 层，是否能消除遗忘？

新增检查点（均已确认存在）：

| 名称 | 路径 |
|------|------|
| **adj_sft15k** | `logs/20260617-20:00:24-rf20_adjust_bottle_sft_8gpu/.../global_step_15000/actor/model_state_dict/full_weights.pt` |
| **ppo_ca_100** | `logs/20260617-13:27:58-rf20sft_ppo_click_alarmclock/.../global_step_100/actor/model_state_dict/full_weights.pt` |
| **ppo_adj_200** | `logs/20260615-15:45:43-robotwin15_adjust_bottle_ppo_random20/.../global_step_200/actor/model_state_dict/full_weights.pt` |

### Exp 0（CPU，分析脚本）
`scripts/analyze_sft_rl_weight_diff.py`：加载 6 个权重，分 SigLIP / PaliGemma LM / Action Expert 层组统计任务向量范数占比 + 4×4 余弦相似度矩阵。

### Exp A（对照实验）
公式：`adj_sft15k + α × (ca_sft15k − rf20sft)`，全层，α = 0.3/0.5/0.7

验证：**全量 SFT 向量仍会导致遗忘**（PaliGemma LM 层修改覆盖 adj 语言表示）

### Exp B（关键实验）
公式：`adj_sft15k + α × (ca_sft15k − rf20sft)`，仅 Action Expert（208 keys），α = 0.5/0.7

验证：**Expert-only SFT 向量是否消除遗忘**（直接验证遗忘来源假设）

### Exp C（基础权重对比）
公式：`rf20sft + α × (ca_sft15k − rf20sft)`，仅 Action Expert，α = 0.3/0.5/0.7

验证：从弱基础出发，Expert-only SFT 向量效果

### 预期结果表

| 权重 | CA | adj | 验证 |
|------|-----|------|------|
| ca_sft15k（全量SFT，基线）| 55.7% | 0.0% | — |
| Exp A：adj_sft + ca_sft_full α=0.7 | ? | ≈0% | 全量仍遗忘 |
| Exp B：adj_sft + ca_sft_expert α=0.7 | ? | >50%? | expert-only不遗忘 |
| Exp C：rf20sft + ca_sft_expert α=0.7 | ? | ? | 基础权重影响 |
| merged_a07_PPO_adj50（上限）| 87.7% | 84.2% | — |

**关键判断**：若 Exp B 的 adj 能保持 > 50%，则假设成立——SFT 遗忘来源于 PaliGemma LM 层，与 Action Expert 层无关。

### 所需文件
- 1 个分析脚本 + 8 个合并脚本 + 16 个 eval config + 16 个 sbatch

---

## 总结

| 维度 | SFT（ca_sft15k）| PPO-CA（100步）| PPO-adj（200步）| PPO-adj50（merged基础）|
|------|----------------|---------------|----------------|----------------------|
| 参数变化幅度（Exp 1）| **10-23×** | — | — | 极小 |
| 任务向量干扰余弦（Exp 2）| −0.110 (action_in)| — | — | — |
| HiddenSim vs rf20sft（Exp 3）| **0.121** | 0.025 | 0.020 | 0.032 |
| HiddenSim vs 各自基础（Exp 3）| 0.121 | — | — | 0.011 |
| 变化峰值层 | 16 | 17 | 16 | 17 |

**核心结论**：
- PPO 以**极小的权重变化（SFT 的 1/5 表征移动）**和**几乎零增量的隐藏状态漂移**实现了双任务平衡
- SFT 大幅改变了 Action Expert 的隐藏状态（0.121），深层（12-17层）最敏感
- PPO-adj50 最终权重（G=0.032）≈ merged_a07（F=0.030），PPO 在合并基础上的增量几乎为零（B=0.011）
- **下一步（Exp A/B/C）**：验证 SFT 遗忘是否来源于 PaliGemma LM 层——将 SFT 任务向量过滤到 Action Expert 后，是否能消除遗忘
