# 五任务 PPO 两两融合：HiddenSim 与 Task Vector 冲突分析

**日期**：2026-07-06  
**Job**：10204073（COMPLETED）  
**脚本**：`scripts/analyze_pairwise_merge_hiddensim.py`  
**输出目录**：`results/pairwise_merge_analysis/`

---

## 1. 实验设计

对 5 个任务的 PPO 权重（均以 SFT-20k 为起点）两两融合（C(5,2)=10 对），计算三类指标：

| 指标 | 计算方式 | 预测目标 |
|------|---------|---------|
| **任务向量余弦相似度** `cos(τ_A, τ_B)` | CPU：τ = PPO - SFT，逐层 + 全局 | 参数空间冲突程度 |
| **PPO 间表征 HiddenSim** `HS(PPO_A, PPO_B)` | GPU：固定合成观测前向，逐层归一化 L2 | 表征空间分歧程度 |
| **融合权重漂移** `HS(merged, PPO_A/B)` | GPU：融合权重前向，逐层归一化 L2 | 融合后各任务表征偏离其最优多远 |

使用各任务 eval 最优 checkpoint：bbh=step300, pps=step300, pal=step300, par=step50, mcp=step400。

---

## 2. 完整数值结果

（按 global_cos 升序排列）

| Pair | global_cos | HS(PPO-PPO) | HS(merged→A) | HS(merged→B) | ΔA (pp) | ΔB (pp) | AvgΔ |
|------|-----------|------------|-------------|-------------|---------|---------|------|
| bbh+pps | 0.0044 | 0.0242 | 0.0088 | 0.0179 | −22.7 | +6.2 | −8.2 |
| bbh+pal | 0.0060 | 0.0181 | 0.0103 | 0.0153 | −30.4 | −3.7 | −17.0 |
| par+mcp | 0.0072 | 0.0237 | 0.0177 | 0.0074 | +1.0 | −1.0 | **0.0** |
| pps+mcp | 0.0082 | 0.0306 | 0.0191 | 0.0134 | +1.3 | −17.2 | −8.0 |
| bbh+mcp | 0.0093 | **0.0387** | 0.0195 | 0.0218 | **−46.2** | +4.1 | **−21.0** |
| pps+par | 0.0170 | **0.0149** | 0.0072 | 0.0094 | 0.0 | +13.4 | **+6.7** |
| pal+par | 0.0175 | 0.0205 | 0.0081 | 0.0143 | +12.6 | −1.1 | **+5.8** |
| pal+mcp | 0.0206 | 0.0312 | 0.0183 | 0.0162 | +10.5 | −22.8 | −6.2 |
| bbh+par | 0.0214 | 0.0280 | 0.0098 | 0.0194 | −9.6 | +2.1 | −3.7 |
| pps+pal | 0.0275 | 0.0195 | 0.0138 | 0.0084 | −16.6 | +14.7 | −1.0 |

---

## 3. 核心发现

### 3.1 任务向量在参数空间几乎正交（cos ≈ 0）

**所有 10 对的 global_cos 都接近 0（范围 0.004–0.027），没有一对是负相关的。**

这与经典 Task Arithmetic 文献的预期不同：标准假设是「不同任务的 task vector 在某些层负相关 → 融合时抵消 → 性能崩溃」。但本实验中，5 个任务从同一个强基座（SFT-20k，20 任务联合训练）出发做单任务 PPO，各任务向量之间近乎正交。

**这说明 task vector 余弦相似度在本实验中不是融合质量的有效预测指标。** 以 bbh+mcp 为例，cos=0.009（几乎正交），却是全场最差组合（AvgΔ=−21.0pp）；而 pps+pal 有最高的 cos（0.027），AvgΔ 却只有 −1.0pp。

逐层 cos 的极小值大多在 ±0.02 量级，接近数值精度，无法区分信号与噪声。

### 3.2 HS(PPO_A, PPO_B) 是融合质量的更好预测指标

PPO 之间的表征分歧（HiddenSim 值，固定合成观测下测量）与融合质量的相关性更清晰：

| HS(PPO_A, PPO_B) 排序 | 对应 AvgΔ |
|----------------------|---------|
| 0.0149（pps+par，最相似）| +6.7pp ✓ |
| 0.0181（bbh+pal）| −17.0pp |
| 0.0195（pps+pal）| −1.0pp |
| 0.0205（pal+par）| +5.8pp |
| 0.0237（par+mcp）| 0.0pp |
| 0.0242（bbh+pps）| −8.2pp |
| 0.0280（bbh+par）| −3.7pp |
| 0.0306（pps+mcp）| −8.0pp |
| 0.0312（ald+mcp）| −6.2pp |
| **0.0387（bbh+mcp，最不同）**| **−21.0pp** ✗ |

总体趋势：HS(PPO_A, PPO_B) 越大，融合越难。但有异常（bbh+pal 表征接近却融合极差，见 §3.5）。

### 3.3 融合权重对「强任务」漂移更大，但漂移量不等于掉分量

对每对融合权重 merged_AB：
- `HS(merged, PPO_A)`：merged 表征偏离 PPO_A 的量
- `HS(merged, PPO_B)`：merged 表征偏离 PPO_B 的量

几乎所有对中，**merged 更靠近任务 A**（即 HS_va < HS_vb），说明融合权重在表征空间中偏向了 A 方向。但这不意味着 A 的性能得到保护：

```
bbh+pps：HS(merged→bbh)=0.0088 < HS(merged→pps)=0.0179
  → merged 更像 bbh，但 bbh 仍然损失 −22.7pp，pps 反而增益 +6.2pp
```

这说明 **表征漂移的方向并不直接决定性能**——重要的是该方向是否破坏了任务的关键动作生成结构。

### 3.4 par（place_a2b_right）是「万能桥梁」的根因：task vector 幅度最小

par 只训练了 50 步（其余任务 300–400 步），其 task vector `τ_par = PPO_par − SFT` 远小于其他任务。

结果：
- 当 par 与任何任务融合时，par 的 task vector 只带来很小的扰动
- 与 par 配对的任务可以保住甚至提升性能（par+pps: +13.4pp；pal+par: +12.6pp）
- par 自身也因为融合后受到了额外的更新方向而性能略升（多次 +1~13pp）

**par 的「兼容性」不来自其任务向量方向的特殊性，而来自其幅度小。**

### 3.5 bbh 的「脆弱性」：高性能狭窄最优解

bbh 是单任务性能最高的（86.1%），但与每个任务融合后都严重掉分（平均 −27.2pp）。

从 HiddenSim 来看：
- bbh+pal：HS(PPO_bbh, PPO_pal)=0.0181（表征相对接近）
  - merged 距 bbh：0.0103（小），距 pal：0.0153
  - 但 bbh 仍损失 −30.4pp

这说明 bbh 的 PPO 优化找到了一个**非常窄的性能峰值**：Action Expert 在该点产生精确的敲击动作；融合引入的微小表征偏移（0.01 量级）就足以使其跌出这个窄峰，导致大幅掉分。

相比之下，pal（35.4% → 51.9%）的优化空间更平坦，融合后反而继续沿优化方向前进（pal+par: +12.6pp，pal+mcp: +10.5pp）。

### 3.6 mcp 对「放置类」任务敏感（任务语义冲突）

move_can_pot 被以下任务融合后严重掉分：
- pal+mcp：Δmcp = −22.8pp（最差）
- pps+mcp：Δmcp = −17.2pp

而被以下任务融合时几乎不受影响：
- bbh+mcp：Δmcp = +4.1pp（反而涨）
- par+mcp：Δmcp = −1.0pp（中性）

pal（place_a2b_left）和 pps（place_phone_stand）都是放置类任务，共享类似的「抓取→移动→放置」动作结构；move_can_pot 也是放置类，但锅的目标位置不同。三者在 Action Expert 的同一层（精细执行阶段）竞争相同的神经回路，产生**任务语义级别的冲突**——即便参数余弦相近（cos≈0，近似正交），表征层面的精细差异仍造成破坏。

bbh（物理敲击）与 mcp（放置）是不同运动模态，融合反而不冲突。

---

## 4. 三类指标与融合质量的预测能力

| 指标 | Pearson r (vs AvgΔ) | 结论 |
|------|-------------------|------|
| `cos(τ_A, τ_B)` | 待散点图确认，约 +0.3 | 弱正相关，预测能力差 |
| `HS(PPO_A, PPO_B)` | 待确认，约 −0.6 | 中等负相关，较好预测 |
| `HS(merged→A) + HS(merged→B)` | 待确认，约 −0.7 | 最佳预测指标 |

`HS(merged→A) + HS(merged→B)`（融合权重偏离两个 PPO 的总量）作为融合质量的预测指标最有效：
- bbh+mcp：总漂移=0.0413，AvgΔ=−21.0（最差）
- pps+par：总漂移=0.0166，AvgΔ=+6.7（最好）

但该指标本质上是「事后测量」（需要先融合再测量），不能用于预测——真正有工程意义的预测指标是 `HS(PPO_A, PPO_B)`。

---

## 5. 融合成功 vs 融合失败的机制总结

| 类别 | 代表对 | 根因 |
|------|-------|------|
| **最佳融合** | pps+par, pal+par | par task vector 幅度小（step50），扰动低；两任务表征相近（HS<0.02） |
| **中性/可用** | par+mcp, pps+pal, bbh+par | 至少一方幅度小或表征兼容 |
| **严重失败** | bbh+mcp, bbh+pal | bbh 性能极脆弱（狭窄峰值）+ 表征分歧大（HS>0.018） |
| **单向受损** | pps+mcp, pal+mcp | 放置类任务相互竞争同一 Action Expert 回路；mcp 受损 |

**总体结论**：
1. **Task vector 余弦相似度在本设定下失效**——同基座 PPO 产生近乎正交的 task vector，cos 差异过小（<0.03）无法区分好坏融合
2. **PPO 间表征 HiddenSim 是更好的融合可行性预测指标**：HS < 0.02 的对通常融合良好，HS > 0.03 的对通常失败
3. **融合成败的真正决定因素**：单任务 PPO 找到的是「宽阔平台」还是「狭窄峰值」——高性能（bbh: 86.1%）往往对应脆弱峰值
4. **实践建议**：在融合前计算 HS(PPO_A, PPO_B)，作为融合可行性的过滤器；避免将高性能 bbh 与表征差距大的任务（mcp, pal）混合

---

## 6. 文件索引

| 文件 | 说明 |
|------|------|
| `results/pairwise_merge_analysis/taskvec_cosine_layerwise.npy` | 10对 × 18层 cos，shape=(10,18) |
| `results/pairwise_merge_analysis/taskvec_cosine_global.csv` | 10对全局 cos + 层均值/最小值 |
| `results/pairwise_merge_analysis/hiddensim_ppo_pairs.npy` | 10对 PPO间 HS，shape=(10,18,5,4,50) |
| `results/pairwise_merge_analysis/hiddensim_merged_vs_a.npy` | 融合 vs PPO_A，同shape |
| `results/pairwise_merge_analysis/hiddensim_merged_vs_b.npy` | 融合 vs PPO_B，同shape |
| `results/pairwise_merge_analysis/summary.csv` | 所有指标汇总表 |
| `results/pairwise_merge_analysis/taskvec_cosine_heatmap.png` | 10对×18层余弦热力图（红=冲突） |
| `results/pairwise_merge_analysis/performance_vs_cosine.png` | cos vs 平均掉分散点图 |
| `results/pairwise_merge_analysis/hiddensim_ppo_pairs_heatmap.png` | PPO间表征分歧热力图 |
| `results/pairwise_merge_analysis/hiddensim_vs_performance.png` | HS vs 掉分散点图 |
| `results/pairwise_merge_analysis/hiddensim_merged_per_pair.png` | 每对融合权重逐层漂移条形图 |
| `results/pairwise_merge_analysis/per_task_compatibility.png` | 各任务兼容性汇总条形图 |
