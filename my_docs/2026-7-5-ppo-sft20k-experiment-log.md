# π0.5 (MoE) PPO 训练实验日志 — RoboTwin Tasks

**基础模型**：`pi05_aloha_moe_demo_clean` / SFT-20k checkpoint  
**框架**：RLinf · JAX flow-matching VLA  
**开始日期**：2026-07-02  
**更新日期**：2026-07-06 11:20

---

## SFT-20k → PPO 涨点汇总（截至 2026-07-06）

| 任务 | SFT-20k eval | PPO 最优 eval | 涨点 | 当前状态 | 权重 |
|------|-------------|--------------|------|---------|------|
| beat_block_hammer | 57.8% | **86.1%** @step299 | **+28.3pp** ⭐ | 中止于 step335（job超时）| step100/200/300，57G |
| place_phone_stand | 39.6% | **70.7%** @step299 | **+31.1pp** | 中止于 step450（手动 scancel）| step50-450，152G |
| place_a2b_left | 35.4% | **51.9%** @step299 | **+16.5pp** | 中止于 step~423（手动 scancel）| step50-400，133G |
| move_can_pot | 52.6% | **63.6%** @step399 | **+11.0pp** | 完成 step500（2026-07-06）| step50-500，190G |
| place_a2b_right | 55.1% | **57.1%** @step49 | **+2.0pp** | 运行中 step~166 | step50-150，57G |
| place_object_stand | 68.6% | **65.9%** @step299 | **−2.7pp** | 中止于 step319（job超时）| step100/200/300，57G |
| handover_block | 21.9% | 37.5% @step49 (=SFT) | **−8.3pp** | 放弃（v1崩溃，v2异常） | step350，19G |
| put_bottles_dustbin | — | — | — | 放弃（全零奖励 step96，已 scancel）| 无 |

> eval 使用 temperature=0.6 + manifest seeds（固定）；train 使用 temperature=1.0 + success_seeds。  
> SFT baseline 来自 `2026-6-24-moe-demo-clean-experiment-log.md` 第6节，使用同一 manifest_96ep 评测协议。  
> move_can_pot 最优 eval @step399 为 63.6%，建议对 step400/450/500 做 eval_rollout_epoch: 3 取平均确认最终结果。  
> put_bottles_dustbin：SFT-20k 对此任务几乎无成功能力（step96 train success_once=0%），全零奖励下 PPO 无法收敛，放弃。

---

## 当前运行状态

| 任务 | Job ID | TB Step | train succ | eval succ | 状态 |
|------|--------|---------|-----------|----------|------|
| move_can_pot | 10195106 | **500/500** | — | 63.6% @399 | ✅ 完成（2026-07-06 03:31）|
| place_a2b_left | 10197179 | ~423/500 | 49.2% | 51.9% @299 | ⛔ 手动 scancel（2026-07-06）|
| place_phone_stand | 10197180 | 450/500 | 69.9% | 70.7% @299 | ⛔ 手动 scancel（2026-07-06 12:20）|
| place_a2b_right | 10200922 | 166/500 | 52.3% | 57.1% @49 | ✅ 正常 |
| put_bottles_dustbin | 10201671 | 96/500 | 0% | — | ❌ 放弃（全零奖励，已 scancel）|

---

## 实验详细记录

### 1. move_can_pot PPO

- **Job**：10195106
- **启动**：2026-07-03 00:38 · 节点 SH-IDCA1404-10-140-54-25
- **配置**：标准 v1

| 指标 | 值 | Step |
|------|----|------|
| env/success_once | 60.9% | 459 |
| eval/success_once（最优） | **63.6%** | 399 |
| eval/success_once（最新） | 51.9% | 449 |
| approx_kl | 0.037 | 399 |
| value_loss | 0.061 | 399 |

**涨点**：SFT 52.6% → PPO 最优 eval 63.6%（**+11.0pp**）

**⚠️ Log Worker 死亡**：`run_embodiment.log` 在 step ~90 停止更新，TensorBoard 持续活跃。这是 RLinf 已知模式：Ray logging actor 死亡不影响训练主循环。

**⚠️ eval 方差**：单次 eval 仅 32 episodes，±10pp 波动正常。step 399 eval=63.6% vs step 449 eval=51.9%，不代表策略退化。建议训练结束后对 step 400/450/500 做 `eval_rollout_epoch: 3` 取平均。

**权重**：`global_step_50 ~ global_step_450`（9个，152G）均保存。

---

### 2. place_a2b_left PPO

- **Job**：10197179
- **启动**：2026-07-03 21:07 · 节点 SH-IDCA1404-10-140-54-2
- **配置**：标准 v1

| 指标 | 值 | Step |
|------|----|------|
| env/success_once | 48.4% | 327 |
| eval/success_once | **51.9%** | 299 |
| approx_kl | 0.029 | 266 |
| policy_loss | −0.075 | 266 |
| value_loss | 0.075 | 266 |

**涨点**：SFT 35.4% → PPO eval 51.9%（**+16.5pp**）

**权重**：`global_step_50/100/150/200/250/300`（6个，114G）均保存。

---

### 3. place_phone_stand PPO

- **Job**：10197180
- **启动**：2026-07-03 21:07 · 节点 SH-IDCA1404-10-140-54-39
- **配置**：标准 v1

| 指标 | 值 | Step |
|------|----|------|
| env/success_once | 71.9% | 344 |
| eval/success_once | **70.7%** | 299 |
| approx_kl | 0.029 | 278 |
| policy_loss | −0.102 | 278 |
| value_loss | 0.073 | 278 |

**涨点**：SFT 39.6% → PPO eval 70.7%（**+31.1pp**）

**观察**：step 249 时 eval 仅 33.3%（vs train 62.9%，gap 约 30pp），step 299 eval 跳升至 70.7%，gap 收窄。说明早期 eval 方差大，并非过拟合。

**权重**：`global_step_50/100/150/200/250/300`（6个，114G）均保存。

---

### 4. handover_block PPO v1（失败/放弃）

- **Job**：10192376（已 scancel）
- **启动**：2026-07-02 10:48 · 手动取消于 step 350

**结果**：PPO 低于或等于 SFT baseline，放弃该任务 PPO 实验。

| Checkpoint | eval/success_once | 备注 |
|------------|-------------------|------|
| SFT-20k (init) | 21.9% | SFT baseline（6月24日测） |
| SFT-20k eval @step0（PPO init eval） | **37.5%** | PPO训练前测的baseline（manifest seeds） |
| step 350 最终 eval | **29.2%** | 独立 eval Job 10200319 确认 |

**注**：PPO init eval (37.5%) 与 6月24日 SFT eval (21.9%) 差异较大，原因可能是评测种子/episode_steps 不同。PPO 训练结果 29.2% 低于 init eval 37.5%，确认训练有害。

**失败原因**：
1. 稀疏奖励（训练 success_once 初始 ~9%），91% 轨迹零奖励
2. kl_beta=0.0，无 SFT 正则化
3. lr=5e-6 过高，策略振荡
4. rollout_epoch=4，每步数据量不足

**handover_block v2（已放弃）**：
- Job 10200331：filter_rewards bug（chunk-level reward ~0.0002 << lower_bound 0.1，100% episode 被过滤，训练指标全零）
- 修复后 Job 10201021：`ValueError: Unsupported object type`（collective 通信层内存损坏），多个 RolloutGroup rank 同时报错，无 checkpoint 保存
- Job 10201404：立即 scancel，任务放弃

**权重**：仅保留 `global_step_350`（19G）。

---

### 5. place_a2b_right PPO

- **Job 10200336**（Gloo timeout crash）→ **Job 10200922**（当前）
- **启动（当前）**：2026-07-05 10:45 · 节点 SH-IDCA1404-10-140-54-47
- **配置**：标准 v1

| 指标 | 值 | Step |
|------|----|------|
| env/success_once | 52.0% | 73 |
| eval/success_once | **57.1%** | 49 |
| approx_kl | 0.014 | 8 |

**涨点**：SFT 55.1% → PPO eval 57.1%（**+2.0pp**，尚早期，仅供参考）

**权重**：`global_step_50`（1个，19G）。

---

### 6. put_bottles_dustbin PPO

- **Job**：10201671（2026-07-05 21:20 提交）
- **节点**：待分配
- **配置**：`moe_sft20k_put_bottles_dustbin_ppo100.yaml`（micro_bs=64, global_bs=2048, lr=5e-6）

**注意**：config 中 `eval.seeds_path: null`，训练期间不会产生 eval/success_once 指标，只有 env/success_once（train 采样）。如需 eval 需单独补充 manifest。

**权重**：无（刚启动）。

---

### 7. beat_block_hammer PPO（2026-06-25，已停止）

- **Job**：运行于 2026-06-25 11:37，log dir `logs/20260625-11:37:02-moe_sft20k_beat_block_hammer_ppo100`
- **配置**：`moe_sft20k_beat_block_hammer_ppo100.yaml`，base model：SFT-20k
- **停止原因**：job 超时，中止于 step 335/500

| 指标 | 值 | Step |
|------|----|------|
| eval/success_once（最优） | **86.1%** | 299 |

**涨点**：SFT 57.8% → PPO eval 86.1%（**+28.3pp** ⭐ 全任务最高）

**权重**：`global_step_100/200/300`（3个，57G）保存。step 300 为最优点，可直接用于 eval 或续训。

**建议**：step 300 → 500 段尚未覆盖，可考虑从 step 300 checkpoint 续训至 500。

---

### 8. place_object_stand PPO（2026-06-25，已停止）

- **Job**：运行于 2026-06-25 11:47，log dir `logs/20260625-11:47:20-moe_sft20k_place_object_stand_ppo100`
- **配置**：`moe_sft20k_place_object_stand_ppo100.yaml`，base model：SFT-20k
- **停止原因**：job 超时，中止于 step 319/500

| 指标 | 值 | Step |
|------|----|------|
| eval/success_once（最优） | **65.9%** | 299 |

**涨点**：SFT 68.6% → PPO eval 65.9%（**−2.7pp**，与 SFT 持平，PPO 无明显提升）

**权重**：`global_step_100/200/300`（3个，57G）保存。

**分析**：SFT baseline 已达 68.6%（中高水平），PPO 未能进一步提升；可能需要更长训练（step 300 未收敛）或超参调整。

---

## 权重保存情况

| 任务 | 保存的 checkpoints | 总大小 | 备注 |
|------|-------------------|--------|------|
| beat_block_hammer | step 100/200/300 | **57G** | 已停止，step 300 最优 |
| place_phone_stand | step 50/100/150/200/250/300 | **114G** | 训练中，step~344 |
| place_a2b_left | step 50/100/150/200/250/300 | **114G** | 训练中，step~327 |
| move_can_pot | step 50/100/150/200/250/300/350/400/450 | **152G** | 训练中，step~459 |
| place_a2b_right | step 50 | **19G** | 训练中，step~73 |
| place_object_stand | step 100/200/300 | **57G** | 已停止，step 300 |
| handover_block | step 350（仅此一个） | **19G** | 已放弃 |
| put_bottles_dustbin | — | — | 刚启动 |
| **合计** | | **~532G** | |

---

## 关键发现

### 1. TensorBoard 在 log worker 死亡后继续写入
Ray logging actor（写 `run_embodiment.log`）可独立于训练循环死亡。TensorBoard event writer 在独立 actor 中运行，不受影响。**判断训练是否卡住：查 TB mtime + 文件大小，而非 log mtime。** 实证：move_can_pot log 死于 step 90，TB 持续活跃至 step 459+。

### 2. filter_rewards lower_bound 必须匹配奖励量级
chunk_level reward 在稀疏任务中约为 ~0.0002/episode。`rewards_lower_bound: 0.1` 会静默过滤所有数据，表现为训练指标全 0（无报错）。稀疏任务建议 `filter_rewards: False`，或将 lower_bound 设为 ~1e-4。

### 3. 稀疏奖励任务（初始 success < 15%）不适合标准 v1 超参
handover_block 初始 train success ~9%，PPO 训练失败。建议使用 v2 超参：lr=2e-6，kl_beta=0.02，rollout_epoch=8，global_bs=2048。即便如此，handover_block 仍因 collective 通信异常多次崩溃，最终放弃。

### 4. eval 单次样本量小，方差大
32 envs × 1 eval_rollout_epoch = 32 episodes/eval，±10pp 波动正常（place_phone_stand step249: 33.3% vs step299: 70.7%）。训练结束后务必用 `eval_rollout_epoch: 3`（96 episodes）取平均再下结论。

### 5. eval episode steps = 400（已验证）
通过 click_alarmclock 消融确认。200-step eval 会低估 success rate。所有配置统一 `max_episode_steps: 400`。

### 6. Gloo broadcast timeout 为节点级偶发故障
place_a2b_right Job 10200336 因 Gloo timeout 崩溃，重提后正常。非代码或配置问题，属 NCCL/Gloo 通信偶发异常。

---

## 配置参考

### 标准 v1（place_a2b_left / place_phone_stand / move_can_pot / place_a2b_right）

```yaml
actor:
  micro_batch_size: 32
  global_batch_size: 1024
  optim:
    lr: 5.0e-06
algorithm:
  rollout_epoch: 4
  kl_beta: 0.0
  filter_rewards: False
  update_epoch: 5
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
env:
  max_episode_steps: 400
val_check_interval: 50
save_interval: 50
```

### 升级 v2（稀疏奖励任务，如 handover_block）

```yaml
actor:
  micro_batch_size: 64
  global_batch_size: 2048
  optim:
    lr: 2.0e-06
algorithm:
  rollout_epoch: 8
  kl_beta: 0.02
  filter_rewards: False
```

**公共设置**：`model_path = pi05_aloha_moe_demo_clean/.../20000`，eval seeds = `*_manifest_96ep.json`
