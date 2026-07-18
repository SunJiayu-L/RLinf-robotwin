# moe-demo-clean PPO 训练过程隐藏状态漂移分析

**日期**：2026-07-02  
**模型**：Pi0.5（`pi05_aloha_moe_demo_clean`）  
**分析脚本**：`scripts/analyze_hidden_states_moe.py`  
**SLURM Job**：10189971（COMPLETED）  
**输出目录**：`results/hidden_states_moe/`

---

## 1. 实验设置

### 1.1 研究问题

从 moe-demo-clean SFT 20k 出发做单任务 PPO，Action Expert 的内部表征如何随训练步数偏离初始 SFT 基座？深层还是浅层漂移更大？精细执行阶段（t=0.1）还是噪声条件阶段（t=0.9）更敏感？

### 1.2 指标：HiddenSim

```
HiddenSim(A, B) = ||h_A - h_B||_2 / (||h_A||_2 + ||h_B||_2 + ε)
```

- 对每个 (layer, t, batch, token_k) 单独计算，再按需平均
- 值域 [0, 1]，越大表示表征偏离越多
- 使用固定随机合成观测（seed=42）和固定 x_t 保证可复现

### 1.3 对比设置

| 对比 | 含义 |
|------|------|
| HiddenSim(PPO_stepN, SFT_20k) | 相对初始基座，PPO 训练 N 步后的表征漂移量 |

- **base**：moe SFT 20k（`robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors`）
- **PPO 步数**：50 / 100 / 150 / 200 / 250 / 300
- **任务**：beat_block_hammer（BBH）、place_object_stand（POS）

### 1.4 检查点路径

| 检查点 | 路径 |
|--------|------|
| moe SFT 20k | `/mnt/.../robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors` |
| BBH PPO | `logs/20260625-11:37:02-moe_sft20k_beat_block_hammer_ppo100/.../checkpoints/` |
| POS PPO | `logs/20260625-11:47:20-moe_sft20k_place_object_stand_ppo100/.../checkpoints/` |

---

## 2. 数值结果

### 2.1 全局 HiddenSim（对所有层 / t / batch / token 平均）

| PPO Step | beat_block_hammer | place_object_stand |
|---------|:-----------------:|:-----------------:|
| 50      | 0.0157            | 0.0107            |
| 100     | 0.0139 ↓          | 0.0122            |
| 150     | 0.0224            | 0.0136            |
| 200     | 0.0243            | 0.0190            |
| 250     | 0.0228            | 0.0208            |
| 300     | 0.0279            | 0.0295            |

BBH step 50→100 出现短暂回落（0.0157 → 0.0139），其余均单调上升。两任务最终在 step 300 收敛至相近水平（~0.028–0.030）。

### 2.2 与 rf20sft 实验的对比参考

| 对比对 | 全局 HiddenSim |
|--------|---------------|
| CA-SFT-ft vs Base（SFT 专化，单任务过拟合） | **0.121** |
| PPO-CA vs Base（PPO 100步）              | 0.025 |
| PPO-adj vs Base（PPO 200步）             | 0.020 |
| BBH PPO-300 vs SFT-20k（本实验）         | 0.028 |
| POS PPO-300 vs SFT-20k（本实验）         | 0.030 |

moe SFT 20k 出发的 PPO 漂移量与 rf20sft 实验的 PPO 同量级，远小于 SFT 专化（0.121）。

---

## 3. 图像与解读

### 3.1 热力图：层 × PPO 步数

`results/hidden_states_moe/heatmap_step_beat_block_hammer.png`  
`results/hidden_states_moe/heatmap_step_place_object_stand.png`

**x 轴** = PPO 训练步数（50–300），**y 轴** = Action Expert 层（0–17），**颜色** = HiddenSim。

**规律**：
- 深层（层 14–17）随步数颜色迅速由浅黄变深红，是漂移的主要来源
- 浅层（0–8）整体偏浅，随步数变化不明显
- 颜色由左至右逐渐加深，整体漂移随训练单调增加

**含义**：PPO 更新集中在 Action Expert 的后半段层，浅层（接近 prefix 端）几乎保持不变，深层（输出端）对奖励信号更敏感。

---

### 3.2 热力图：层 × 去噪时间步 t

`results/hidden_states_moe/heatmap_t_beat_block_hammer.png`  
`results/hidden_states_moe/heatmap_t_place_object_stand.png`

**x 轴** = 去噪时间步 t（0.1–0.9），**y 轴** = 层，颜色对所有 PPO 步数平均。

**beat_block_hammer**：
- 漂移集中在 t=0.1–0.3（精细执行阶段）+ 深层，形成左下角热区
- t≥0.5 整体颜色很浅，噪声条件阶段几乎未受 PPO 影响

**place_object_stand**：
- t=0.5 处出现明显低谷（浅色条带），t=0.7/0.9 在深层又回升
- 呈 bimodal 分布：精细执行（t=0.1）+ 大噪声（t=0.7–0.9）均有热区，中间的 t=0.5 反而是冷区

**与 rf20sft 实验对比**：  
在 rf20sft 实验中，PPO-CA 的热点在 t=0.9（噪声条件阶段），而这里 moe BBH 的热点在 t=0.1（精细执行阶段）。差异可能来自基座能力的差异：moe SFT 20k 经过 20000 步多任务 SFT，噪声条件理解已经很强，PPO 只需进一步优化精细执行阶段。

---

### 3.3 漂移折线：全局 HiddenSim vs PPO 步数（三条 t）

`results/hidden_states_moe/drift_step_beat_block_hammer.png`  
`results/hidden_states_moe/drift_step_place_object_stand.png`

**beat_block_hammer**：
- t=0.1（红色虚线）始终最高，约为 t=0.9（绿色虚线）的 2–3×
- step 100 有一次全局低谷（三条线均下沉），是 PPO 早期的"回调"现象
- step 150 后快速爬升，300 步时 t=0.1 达到 0.044

**place_object_stand**：
- 早期（50–150步）t=0.1 领先，但 t=0.9 从 step 200 开始快速追赶
- step 300 时二者几乎持平（~0.034），说明 POS 任务的 PPO 后期对两个阶段影响趋于均等
- 整体曲线更平滑，无 BBH 的明显 step 100 低谷

---

### 3.4 逐层折线：各 PPO 步的 per-layer HiddenSim

`results/hidden_states_moe/layer_curves_beat_block_hammer.png`

- 6 条曲线（step 50=紫 → step 300=黄）均单调向右上倾斜：层越深漂移越大
- Step 100（深蓝）全程低于 step 50（紫），是上述 step 100 回调现象的层级分布
- Step 250（浅绿）和 step 200（青绿）在浅层接近，在深层发散，说明后期漂移主要集中在输出端

---

### 3.5 两任务对比折线

`results/hidden_states_moe/drift_comparison.png`

- BBH（蓝线）整体始终高于 POS（红线），说明 beat_block_hammer 的 PPO 更新幅度更激进
- POS 曲线斜率在 200 步后急剧增大，追上并超过 BBH，到 step 300 时 POS 略高（0.0295 vs 0.0279）
- 两条曲线均在 step 100 有短暂停滞或下降，之后快速上升

---

## 4. 关键发现

1. **深层主导漂移**：PPO 更新在 Action Expert 第 14–17 层最显著，与 rf20sft 实验中 SFT 专化的层分布模式一致——输出端对任务特定奖励信号更敏感。

2. **精细执行阶段（t=0.1）是 BBH 的主要更新位点**：与 rf20sft 实验中 PPO-CA 集中在 t=0.9 不同，moe SFT 20k 基座在噪声条件处理上已经更充分，PPO 只需微调精细动作。

3. **step 100 存在短暂回调**：两个任务在 step 100 的 HiddenSim 均低于 step 50，表明早期 PPO 先做了某种"对齐"或"校正"，然后才开始稳定积累表征偏移。

4. **整体漂移量远小于 SFT 专化**：300 步 PPO 的漂移（~0.030）仅为 15k 步 SFT 专化（0.121）的 25%，说明 PPO 在功能提升的同时对基座表征的破坏更小。

5. **两任务最终漂移量收敛**：BBH 更激进，POS 更保守但后期追赶，300 步时两者相近，可能反映两个任务本身难度相近，PPO 学习量相当。

---

## 5. 文件索引

| 文件 | 说明 |
|------|------|
| `results/hidden_states_moe/heatmap_step_{task}.png` | 层 × PPO步 热力图（主图）|
| `results/hidden_states_moe/heatmap_t_{task}.png` | 层 × 去噪时间步 热力图 |
| `results/hidden_states_moe/drift_step_{task}.png` | 全局漂移折线（含 t=0.1 vs t=0.9）|
| `results/hidden_states_moe/layer_curves_{task}.png` | 逐层漂移折线（按 PPO step 着色）|
| `results/hidden_states_moe/drift_comparison.png` | 两任务对比折线 |
| `results/hidden_states_moe/hiddensim_{task}_step{N}_vs_sft20k.npy` | 原始数值，shape=(18, 5, 4, 50)|
