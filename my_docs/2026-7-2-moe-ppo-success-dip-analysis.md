# moe-demo-clean PPO 成功率下降现象分析

**日期**：2026-07-02
**任务**：beat_block_hammer（BBH）、place_object_stand（POS）
**现象**：两个任务在 PPO step 100→150 附近出现成功率明显下降，step 200 后逐渐恢复
**分析图**：`results/hidden_states_moe/training_metrics_dip_analysis.png`

---

## 1. 现象概述

### 1.1 成功率变化

| PPO Step (eval) | BBH success_once | POS success_once |
| --------------- | :--------------: | :--------------: |
| 49              |     **0.823**    |     **0.706**    |
| 99              |       0.788      |       0.743      |
| 149             |     0.532 ↓↓     |     0.510 ↓↓     |
| 199             |       0.531      |       0.500      |
| 249             |       0.563      |      0.583 ↑     |
| 299             |   **0.861** ↑↑   |      0.659 ↑     |

两个任务都在 step 149/199 附近跌至约 50%。随后 BBH 明显恢复，并最终超过初始成功率；POS 也恢复，但恢复幅度较慢，最终未超过初始成功率。

---

## 2. Trajectory 与 Step 的关系

在本实验中，需要区分两个层级：

```text
trajectory = 一整条 episode / rollout 轨迹
step       = trajectory 中的一次环境交互
```

一条 trajectory 可以表示为：

```text
τ = {
  obs_0, action_0, reward_0,
  obs_1, action_1, reward_1,
  ...
  obs_T
}
```

其中每一个 `obs_t, action_t, reward_t` 对应一个 environment step。

对于 RoboTwin / VLA 任务：

```text
一次完整的机器人任务执行 = 一条 trajectory
机器人每执行一次 action chunk / 环境推进一次 = 一个 environment step
flow matching 内部 denoising step = 模型内部动作生成步，不是环境 step
```

因此：

* `success_once` 是 **trajectory-level** 指标，表示整条轨迹是否曾经成功。
* `returns / values / advantages / logprobs` 通常是 **step-level** 指标。
* PPO 的 actor 更新主要发生在 step-level：每个 step 的 logprob 会乘以对应的 advantage。
* 但成功与失败的语义通常来自 trajectory-level success 标志。

因此，如果 rollout batch 中没有保存每条 trajectory 的 `success_once`，就无法精确计算：

```python
value_success_mean
value_failure_mean
adv_success_mean
adv_failure_mean
corr(value, success)
corr(advantage, success)
```

这也是当前分析中的一个主要限制。

---

## 3. Critic Explained Variance 现象

### 3.1 Explained Variance 数值

从 TensorBoard 提取的 50 步窗口均值如下：

```text
窗口          BBH              POS
  0–49       −0.174           −0.409
 50–99       −1.427           −0.895
100–149      −0.985           −1.240
150–199      −1.378           −0.866
200–249      −1.142           −2.168
250–299      −1.610           −2.396
```

Explained variance 的定义为：

```text
EV = 1 - Var(returns - values) / Var(returns)
```

含义是：

```text
EV 接近 1：critic 能较好解释 return 的变化
EV 接近 0：critic 接近常数 baseline
EV < 0：critic 的 residual 方差比 returns 本身还大，说明 critic 作为 baseline 质量较差
```

需要注意：

```text
EV < 0 不等价于 critic 与 return 负相关。
```

EV 为负只说明：

```text
Var(returns - values) > Var(returns)
```

也就是 residual 的方差过大。造成 EV 为负的原因可能包括：

1. value 与 return 确实负相关；
2. value 与 return 正相关，但 value 尺度过大；
3. value 噪声过大；
4. returns 方差很小，导致 EV 对噪声极其敏感；
5. reward / done / bootstrap 时间步没有对齐；
6. trajectory-level success 与 step-level value 没有正确对应；
7. critic 在非平稳 policy 分布下追逐移动目标。

因此，不能仅凭 EV < 0 推断“critic 在提供反向梯度信号”。

---

## 4. Rollout Advantages Mean

50 步窗口均值如下：

```text
窗口          BBH              POS
  0–49       −0.511           −0.211
 50–99       −0.736           −0.211
100–149      −0.711           −0.216
150–199      −0.901           −0.217
200–249      −0.836           −0.197
250–299      −0.964           −0.189
```

现象：

* BBH 的 advantage mean 从约 −0.51 下滑到 −0.96，说明 critic 对 return 存在较明显的整体高估或 calibration bias。
* POS 的 advantage mean 稳定在 −0.2 左右，说明其整体高估幅度相对 BBH 较小。
* advantage mean 长期为负，说明：

```text
mean(values) > mean(returns)
```

如果 `advantages_mean` 是 pre-normalization advantage，则可以反推：

```python
implied_value_mean = returns_mean - advantages_mean
```

但需要注意：

```text
如果 TensorBoard 中记录的是 normalized advantage，
则该反推不成立。
```

---

## 5. 其他 PPO 指标

| 指标            | BBH 范围        | POS 范围        | 解读                   |
| ------------- | ------------- | ------------- | -------------------- |
| approx_kl     | 0.040–0.051   | 0.040–0.048   | 无明显突变                |
| clip_fraction | 0.152–0.168   | 0.149–0.167   | 稳定，略高但无异常            |
| grad_norm     | 28.1–29.2     | 25.9–28.0     | 无梯度爆炸                |
| value_loss    | 0.063–0.090   | 0.053–0.076   | 整体下降，critic loss 在优化 |
| policy_loss   | −0.098→−0.121 | −0.113→−0.121 | 正常范围                 |

这些指标没有显示明显的 PPO 更新爆炸。因此，成功率下降不太像是由单次 KL 过大、梯度爆炸或 clip fraction 突变直接导致。

更可能的原因是：

```text
critic/value baseline 质量较差
        +
advantage 估计存在偏差或排序噪声
        +
策略逐渐偏离 SFT 有效行为区域
```

---

## 6. 对原始假说的修正

### 6.1 原始假说

原先的解释是：

```text
EV < 0
→ critic 与 return 负相关
→ 成功轨迹被打低分，失败轨迹被打高分
→ PPO 收到反向梯度
→ 成功率下降
```

这个解释过强，问题在于：

```text
EV < 0 不能推出 value 与 return 负相关。
```

EV 只说明：

```text
Var(returns - values) > Var(returns)
```

并不能说明 critic 的排序方向一定是反的。

---

### 6.2 现有数据支持的新解释

从 TensorBoard 的 step-level 聚合均值推导：

```python
implied_value_mean = returns_mean - advantages_mean
```

得到跨训练步相关性：

| 指标                                          |    BBH    |    POS    |
| ------------------------------------------- | :-------: | :-------: |
| corr(implied_value_mean, returns_mean) 跨训练步 | **+0.87** | **+0.55** |
| corr(implied_value_mean, success_once) 跨训练步 | **+0.91** | **+0.55** |

这说明：

```text
从训练进程的宏观趋势看，value_mean 和 returns_mean 是正相关的。
```

也就是说，critic 并不是在整体趋势上“倒着预测”。更准确的说法是：

```text
critic 的整体方向并非完全错误，
但 value 的绝对标定和 batch 内 residual 方差存在严重问题。
```

同时，implied value mean 明显高于 returns mean：

```text
BBH：implied_val ≈ 0.73–1.24，returns_mean ≈ 0.22–0.27
POS：implied_val ≈ 0.44–0.46，returns_mean ≈ 0.22–0.27
```

这说明 critic 存在明显的 value 高估。

但需要注意：

```text
单纯的常数高估不会导致 EV 变负。
```

例如：

```text
V = R + constant
```

会让 value loss 变大，但不会显著增加 `Var(R - V)`。真正导致 EV 变负的是：

```text
batch 内 residual = returns - values 的方差过大。
```

因此，更严谨的结论是：

```text
critic 存在明显 calibration bias；
同时 EV 为负说明 residual 方差过大。
这可能来自 value scale 过大、batch 内噪声过大、时间步错位、done mask 问题或策略分布漂移。
```

---

## 7. 成功率下降的修正机制链条

更合理的机制链条如下：

```text
critic 初始化或训练不充分
        ↓
value 预测存在明显 calibration / scale 问题
        ↓
returns - values 的 residual 方差较大
        ↓
explained_variance 长期为负
        ↓
advantage 估计噪声较大，且均值长期为负
        ↓
PPO actor 虽然做了 advantage normalization，
但 batch 内成功/失败样本的相对排序可能阶段性不稳定
        ↓
策略逐渐偏离 SFT 20k 的有效行为区域
        ↓
step 100–200 eval 成功率下降至约 50%
        ↓
随着更多 rollout 数据进入训练，
actor 重新获得可用的相对 advantage 信号，
部分策略能力恢复
        ↓
step 200+ 后成功率逐渐恢复
```

这里的关键不是：

```text
EV 负一定提供了反向梯度。
```

而是：

```text
EV 负说明 critic 作为 baseline 的解释能力较差；
在这种情况下，advantage 的 batch 内排序可能不稳定，
从而造成 actor 更新阶段性偏离有效策略区域。
```

---

## 8. 为什么 value_loss 下降但 EV 仍然为负

`value_loss` 和 `explained_variance` 衡量的是不同东西。

value loss 通常近似为：

```text
E[(returns - values)^2]
```

它关心整体误差大小，包括：

```text
1. residual 的均值偏差
2. residual 的方差
```

而 EV 主要关心：

```text
Var(returns - values) / Var(returns)
```

也就是 residual 的方差相对于 return 方差的大小。

因此，可能出现：

```text
value_loss 下降
但 explained_variance 仍然变差
```

原因可能是：

```text
residual 的平均偏差变小了，
但 residual 的 batch 内方差仍然很大，
甚至相对于 Var(returns) 更大。
```

所以不能仅凭 value_loss 下降判断 critic 已经提供了高质量 baseline。

---

## 9. 为什么 HiddenSim 没有完全捕捉到成功率下降

| 维度 | HiddenSim 测量的内容               | 成功率下降可能相关的真实因素                           |
| -- | ----------------------------- | ---------------------------------------- |
| 对象 | `gemma_expert` 各层隐藏状态 L2 距离均值 | 策略输出动作质量、advantage 排序、critic baseline 质量 |
| 输入 | 固定合成观测，例如随机图像                 | 真实任务分布下的视觉、状态和动作误差                       |
| 度量 | 偏离 SFT 表征的距离大小                | 偏离方向是否破坏任务关键动作                           |
| 盲区 | 不直接包含 `action_out_proj` 的输出质量 | 动作头更新质量直接决定成功率                           |

HiddenSim 可以反映模型表征是否偏离 SFT，但不能直接判断：

```text
1. PPO 梯度方向是否正确
2. action head 是否产生了更差动作
3. 成功轨迹和失败轨迹的 advantage 排序是否正确
4. critic 是否给出了稳定 value baseline
```

因此，HiddenSim 没有明显捕捉成功率下降并不矛盾。

更准确地说：

```text
HiddenSim 是表征漂移指标；
success_once 是任务执行指标；
explained_variance 是 critic baseline 质量指标。
```

三者相关，但不等价。

---

## 10. POS 悖论：EV 持续恶化但成功率上升

### 10.1 现象

| 窗口      | POS explained_variance | POS success_once（窗口末 eval） |
| ------- | ---------------------- | -------------------------- |
| 0–49    | −0.41                  | 0.706                      |
| 50–99   | −0.89                  | 0.743                      |
| 100–149 | −1.24                  | 0.510 ↓                    |
| 150–199 | −0.87                  | 0.500                      |
| 200–249 | −2.17                  | 0.583 ↑                    |
| 250–299 | **−2.40**              | **0.659** ↑                |

step 200 之后，POS 的 EV 从 −2.17 降至 −2.40，但成功率从 0.583 上升至 0.659。

这说明：

```text
EV 不是成功率的单调预测指标。
```

---

### 10.2 为什么不矛盾

PPO actor 更新使用的是 advantage，而不是 EV 本身。

如果在 batch 内满足：

```text
mean(advantage | success) > mean(advantage | failure)
```

那么即使 critic 的 EV 很差，actor 仍然可能获得正确方向的更新。

尤其在成功/失败比较二元的任务中：

```text
成功轨迹 return 较高
失败轨迹 return 较低
```

即使 critic 整体高估 value，只要高估没有破坏成功/失败的相对 advantage 排序，那么：

```text
成功轨迹 advantage 相对更高
失败轨迹 advantage 相对更低
```

经过 advantage normalization 后，PPO 仍然可能继续提升策略。

因此，更合理的解释是：

```text
EV 持续变负说明 critic 作为 baseline 质量差；
但只要标准化后的 advantage 仍然能把成功轨迹排在失败轨迹前面，
actor 仍然可以继续学习。
```

也就是说：

```text
EV 负没有直接导致崩溃；
真正决定 actor 是否进步的是 batch 内 normalized advantage 的相对排序。
```

---

## 11. 当前分析的局限

目前仍缺少最关键的 per-trajectory 日志。

### 11.1 目前能看到的

当前 TensorBoard 可以看到：

```text
returns_mean
advantages_mean
value_loss
explained_variance
success_once
approx_kl
clip_fraction
grad_norm
policy_loss
```

这些多数是 step-level 或窗口级聚合指标。

### 11.2 目前看不到的

目前无法直接计算：

```text
value_success_mean
value_failure_mean
adv_success_mean
adv_failure_mean
corr(value, return) within batch
corr(value, success) within batch
corr(advantage, success) within batch
mean(normalized_adv | success)
mean(normalized_adv | failure)
```

这些才是判断 critic 是否破坏 PPO 更新方向的核心指标。

---

## 12. 建议新增日志

建议在 rollout batch 中保存 trajectory-level success 信息，并 broadcast 到 step-level。

### 12.1 保存 trajectory success

```python
trajectory_success: Tensor[bsz]  # 0/1
```

其中：

```text
bsz = rollout batch 中的 trajectory 数量
```

### 12.2 绑定 step 与 trajectory

如果数据是 `[seq_len, bsz]` 结构，那么可以直接使用：

```python
step_success = trajectory_success.unsqueeze(0).expand(seq_len, -1)
```

### 12.3 新增指标

```python
success = step_success.bool()

metrics["value_success_mean"] = values[success].mean()
metrics["value_failure_mean"] = values[~success].mean()

metrics["adv_success_mean"] = advantages[success].mean()
metrics["adv_failure_mean"] = advantages[~success].mean()

metrics["adv_norm_success_mean"] = adv_norm[success].mean()
metrics["adv_norm_failure_mean"] = adv_norm[~success].mean()

metrics["return_success_mean"] = returns[success].mean()
metrics["return_failure_mean"] = returns[~success].mean()
```

同时建议记录 batch 内相关性：

```python
metrics["corr_value_return"] = corr(values.flatten(), returns.flatten())
metrics["corr_value_success"] = corr(values.flatten(), step_success.float().flatten())
metrics["corr_adv_success"] = corr(advantages.flatten(), step_success.float().flatten())
metrics["corr_advnorm_success"] = corr(adv_norm.flatten(), step_success.float().flatten())
```

其中最关键的是：

```text
adv_norm_success_mean - adv_norm_failure_mean
```

如果该差值为正，并且持续增大，则说明 PPO actor 仍然获得了正确的相对学习信号。

---

## 13. 建议新增 value calibration 指标

为了区分 value bias、value scale 和 residual variance，建议记录：

```python
residual = returns - values

metrics["return_mean"] = returns.mean()
metrics["value_mean"] = values.mean()
metrics["residual_mean"] = residual.mean()

metrics["return_std"] = returns.std()
metrics["value_std"] = values.std()
metrics["residual_std"] = residual.std()

metrics["value_bias"] = values.mean() - returns.mean()
metrics["value_scale"] = values.std() / (returns.std() + 1e-8)

metrics["residual_var"] = residual.var()
metrics["return_var"] = returns.var()
metrics["ev_manual"] = 1 - residual.var() / (returns.var() + 1e-8)
```

这样可以判断 EV 负值主要来自：

```text
1. value 均值整体偏高
2. value 尺度过大
3. value 噪声过大
4. returns 方差过小
5. residual 方差过大
```

---

## 14. 时间步对齐与 done mask 检查

当前代码片段：

```python
# rlinf/algorithms/utils.py
values = torch.cat([values, torch.zeros((1, bsz), ...)], dim=0)  # bootstrap=0
dones = torch.zeros(seq_len + 1, bsz, dtype=torch.bool, ...)
dones[-1] = True   # 强制末尾为 terminal
```

当前判断：

* `num_steps: 5` 是 flow matching 去噪步数，不是 rollout 截断长度。
* GAE 在完整 episode 上计算，而不是只在 5 个 denoising step 上计算。
* 如果每条 episode 都完整结束，那么 `bootstrap = 0` 大体合理。
* 如果存在 mid-episode success，但中间真实 done 没有写入 GAE，则可能污染 returns 和 advantages。
* 如果存在 partial rollout truncation，则 `bootstrap = 0` 可能低估 tail return，理论上应使用 `V(s_{T+1})`。

需要重点检查：

```text
1. env 的真实 done 是否保存到了 rollout buffer
2. success 后是否立即 terminal
3. success 后的 padding / no-op step 是否参与 value loss 和 PPO loss
4. failed episode 是自然 terminal 还是 time-limit truncation
5. dones[-1] = True 是否覆盖了中间真实 done
6. rewards_t、values_t、dones_t 是否在 GAE 中按同一时间步对齐
```

标准 GAE 递推应该在 done 处断开：

```python
delta_t = reward_t + gamma * value_{t+1} * (1 - done_t) - value_t
adv_t = delta_t + gamma * lambda * (1 - done_t) * adv_{t+1}
```

如果 `done_t` 没有正确标记，critic target 和 advantage 都可能被污染。

---

## 15. 哪些指标可以提前预警成功率下降

### 15.1 可靠预警指标

比单看 explained variance 更可靠的是组合指标：

```text
1. explained_variance 长期 < 0
2. value_bias 持续增大
3. value_scale 明显偏离 1
4. residual_var / return_var 持续增大
5. adv_norm_success_mean - adv_norm_failure_mean 下降
6. approx_kl 没有异常，但 success_once 开始下降
```

其中最关键的是：

```text
adv_norm_success_mean - adv_norm_failure_mean
```

因为它直接反映 PPO actor 是否仍然把成功轨迹看得比失败轨迹更好。

---

### 15.2 当前可用的粗略预警

在没有 per-trajectory 日志的情况下，可以暂时使用：

```python
if explained_variance < -0.5 and advantages_mean < -0.5:
    warn("critic baseline may be unreliable; policy update may become unstable")
```

但这只是粗略预警，不能证明 actor 梯度方向已经错误。

更严谨的预警条件应改为：

```python
if explained_variance < -0.5 and adv_norm_success_mean <= adv_norm_failure_mean:
    warn("PPO advantage ranking may be wrong; pause actor update or critic warm-up")
```

---

## 16. 改进方向

| 问题               | 可能改进                                                            |
| ---------------- | --------------------------------------------------------------- |
| critic 初始化不稳定    | 使用 critic-only warm-up，只训练 value head 若干 update                 |
| value 整体高估       | 对 return 做 normalization，或初始化 value head bias 接近 return 均值      |
| value 尺度过大       | 记录并约束 `std(values) / std(returns)`，必要时做 value normalization     |
| EV 长期为负          | 在 EV < 0 且 advantage 排序变差时暂停 actor，只训练 critic                   |
| 稀疏奖励导致 critic 难学 | 尝试 Monte Carlo return、Reinforce++ 或减小 critic 依赖                 |
| done mask 可能错误   | 保存真实 done，并在 GAE 中正确截断                                          |
| success 信息缺失     | 在 Trajectory dataclass 中保存 `success_once`                       |
| HiddenSim 不敏感    | 增加 action head 输出差异、action MSE、success-conditioned hidden drift |

---

## 17. BBH 与 POS 的差异

| 维度                 | BBH                  | POS                                  |
| ------------------ | -------------------- | ------------------------------------ |
| 成功率最低点             | 53.2%（step 149）      | 50.0%（step 199）                      |
| 最终恢复               | **86.1%**，超过初始 82.3% | 65.9%，未超过初始 70.6%                    |
| advantages_mean    | 波动大，约 −0.5 到 −1.0    | 较稳定，约 −0.2                           |
| explained_variance | 长期为负，中后期仍不稳定         | 后期持续恶化至 −2.40                        |
| 恢复情况               | 恢复强                  | 恢复较慢                                 |
| 可能原因               | 击打任务反馈更直接，成功信号更集中    | 放置任务精度要求高，奖励更稀疏，credit assignment 更难 |

BBH 后期能够明显恢复，可能说明其 reward/action 关系更容易被 PPO 重新利用。POS 虽然也恢复，但 EV 后期持续恶化，说明 critic baseline 仍然没有稳定捕捉 batch 内 return 变化。

---

## 18. 最终结论

本实验中，BBH 和 POS 都在 PPO step 100–200 出现成功率下降，但随后逐渐恢复。结合 TensorBoard 指标，最合理的解释是：

```text
critic/value baseline 在训练早期质量较差，
value 存在明显高估和 calibration 问题，
导致 explained_variance 长期为负。
```

但是需要强调：

```text
EV < 0 不等价于 critic 与 return 负相关，
也不等价于 PPO 一定收到反向梯度。
```

当前更稳妥的判断是：

```text
EV 为负说明 critic 作为 baseline 的解释能力较差；
但 PPO actor 是否仍能学习，取决于标准化后的 advantage 是否仍然把成功轨迹排在失败轨迹前面。
```

因此，POS 中出现“EV 持续恶化但成功率上升”并不矛盾。它说明：

```text
critic 的绝对 value 拟合较差，
但 actor 仍可能从 normalized advantage 的相对排序中获得有效学习信号。
```

后续最关键的验证是加入 per-trajectory 日志，直接计算：

```text
value_success_mean vs value_failure_mean
adv_success_mean vs adv_failure_mean
adv_norm_success_mean vs adv_norm_failure_mean
corr(value, return) within batch
corr(advantage, success) within batch
```

只有这些指标才能判断：

```text
critic 是否真的破坏了 PPO 的学习方向。
```

---

## 19. 文件索引

| 文件                                                            | 说明                       |
| ------------------------------------------------------------- | ------------------------ |
| `results/hidden_states_moe/training_metrics_dip_analysis.png` | 8 指标综合分析图，含成功率低谷标注       |
| `logs/20260625-11:37:02-.../tensorboard/`                     | BBH TensorBoard event 文件 |
| `logs/20260625-11:47:20-.../tensorboard/`                     | POS TensorBoard event 文件 |
| `my_docs/2026-7-2-moe-ppo-hiddensim.md`                       | HiddenSim 表征漂移分析文档       |
