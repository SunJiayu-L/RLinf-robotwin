# handover_block PPO + 参考策略 KL 约束

**日期：** 2026-07-18  
**Branch：** `KL`  
**Commit：** `1368154`

---

## 背景

对 `handover_block` 单任务做 PPO 训练时，Loss 中加入与 5-task 联合 PPO 策略的 log-ratio 正则项：

```
L_total = L_PPO + kl_beta * KL_penalty(π_handover_block, π_5task_joint_ppo)
```

> **注意：** 这里的 KL_penalty 是 **sampled log-ratio regularizer**（近似 KL 惩罚），不是严格意义的 D_KL(π_current ∥ π_ref)。样本来自 rollout/old policy（非 current policy），因此是离策略近似；k1 估计器（`logπ - logπ_ref`）可能为负。这是业界 RLHF 实现的标准做法（TRL、OpenRLHF 均如此），PPO 的 clip_ratio 保证 π_old 与 π_current 足够接近时近似合理，但不应视为数学上严格的 KL 散度优化。实现中使用 `low_var_kl`（k3 estimator：`exp(r) - r - 1`，恒非负，方差最低）。

**参考策略（冻结）：** 5-task 联合 PPO（bbh + ca + cb + hb + mpa，24 GPU × 3 节点，step 50→300），使用 step-100 checkpoint：

```
/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/
  20260712-13:41:23-moe_ppo_5task_resume50to300_bbh_ca_cb_hb_mpa/
  moe_ppo_5task_resume50to300_bbh_ca_cb_hb_mpa/
  checkpoints/global_step_100/actor/model_state_dict/full_weights.pt
```

**训练起点：** SFT `moe_sft20k` checkpoint（同 `moe_sft20k_handover_block_ppo100.yaml`）。

---

## 调用链

```
EmbodiedFSDPActor.run_training
  ├─ [NEW] reference forward pass (cpu_weight_swap + no_grad) → ref_logprobs_raw
  ├─ training forward pass → output_dict["logprobs"]
  └─ policy_loss(**kwargs)  ← 新增 ref_logprobs / kl_beta / kl_penalty_type
       └─ preprocess_loss_inputs  ← [NEW] ref_logprobs 做同样的 logprob_type reshape
            └─ compute_ppo_actor_loss  ← [NEW] kl_penalty(logprobs, ref_logprobs) 加入 loss
```

正则项使用 `low_var_kl`（k3）估计器：`exp(ref_log - current_log) - (ref_log - current_log) - 1`，恒非负，方差低于 k1。样本来自 rollout old policy，是近似估计。

---

## 文件改动

### 1. `rlinf/algorithms/utils.py` — `preprocess_loss_inputs`

在三个 `logprob_type` 分支中，对 `ref_logprobs` 做与 `logprobs` 完全相同的 reshape：

| logprob_type | ref_logprobs shape | bsz 含义 |
|---|---|---|
| `token_level` | `[bsz, num_chunks, action_dim]` | 展平后的 action chunk 样本数 |
| `action_level` | `[bsz, num_chunks]`（sum over action_dim）| 同上 |
| `chunk_level`（handover_block 使用） | `[bsz]`（sum over dim=[1,2]）| 同上 |

> **bsz 不是 episode 数。** rollout batch 有额外的 `n_chunk_step` 维度，训练前已展平为多个模型调用样本，所以 bsz 是"展平后的 action chunk 样本数"（一个 episode 产生多条训练样本）。

### 2. `rlinf/algorithms/losses.py` — `compute_ppo_actor_loss`

- 新增 `from rlinf.algorithms.utils import kl_penalty`
- 在 `critic_warmup` 块之后插入 KL 计算：
  ```python
  ref_kl_loss = torch.tensor(0.0, device=logprobs.device)
  _ref_logprobs = kwargs.get("ref_logprobs", None)
  _kl_beta = float(kwargs.get("kl_beta", 0.0))
  if _ref_logprobs is not None and _kl_beta > 0:
      kld = kl_penalty(logprobs, _ref_logprobs, kl_type)
      ref_kl_loss = loss_agg_func(kld, loss_mask, loss_mask_ratio)
      policy_loss = policy_loss + _kl_beta * ref_kl_loss
  ```
- 新增 metrics：`actor/ref_kl_loss`

### 3. `rlinf/workers/actor/fsdp_actor_worker.py` — `EmbodiedFSDPActor.init_worker`

`setup_model_and_optimizer()` 之后加载参考策略权重到 CPU，并做 key 兼容性校验：

```python
self.ref_policy_state_dict = None
self.offload_model_buffer = None
ref_policy_path = self.cfg.algorithm.get("ref_policy_path", None)
if self.cfg.algorithm.kl_beta > 0 and ref_policy_path is not None:
    self.ref_policy_state_dict = torch.load(ref_policy_path, map_location="cpu", ...)
    self.offload_model_buffer = {}
    model_keys = set(self.model.state_dict().keys())
    ref_keys = set(self.ref_policy_state_dict.keys())
    if model_keys != ref_keys:
        missing = list(model_keys - ref_keys)[:5]
        unexpected = list(ref_keys - model_keys)[:5]
        raise RuntimeError(
            f"[KL] ref_policy key mismatch. Missing: {missing}. Unexpected: {unexpected}."
        )
```

> **为何要校验：** `cpu_weight_swap` 内部调 `load_state_dict(strict=True)`，key 不匹配会在第一次 reference forward 时崩溃且报错不直观。外部 `.pt` 文件的 key 格式可能与 FSDP 包装后的 `model.state_dict()` 不一致（FSDP 有时会给 key 加前缀），提前在 `init_worker` 校验可快速定位问题。

### 4. `rlinf/workers/actor/fsdp_actor_worker.py` — `EmbodiedFSDPActor.run_training`

在 training forward 之前插入 reference forward pass：

```python
ref_logprobs_raw = None
if self.cfg.algorithm.kl_beta > 0 and self.ref_policy_state_dict is not None:
    with torch.no_grad():
        with self.amp_context:
            with cpu_weight_swap(self.model, self.ref_policy_state_dict, self.offload_model_buffer):
                _ref_out = self.model(forward_inputs=forward_inputs,
                                      compute_logprobs=True, compute_entropy=False,
                                      compute_values=False, use_cache=False)
        ref_logprobs_raw = _ref_out["logprobs"].detach().float()
```

并在 `kwargs` 中传入：

```python
"ref_logprobs":    ref_logprobs_raw,
"kl_beta":         self.cfg.algorithm.kl_beta,
"kl_penalty_type": self.cfg.algorithm.get("kl_penalty", "kl"),
```

> `cpu_weight_swap` 和 `kl_penalty` 在该文件中已有 import，无需新增。

### 5. `examples/embodiment/config/moe_sft20k_handover_block_ppo_kl.yaml`（新建）

基于 `moe_sft20k_handover_block_ppo100.yaml`，新增两个字段：

```yaml
algorithm:
  kl_beta: 0.01
  ref_policy_path: "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/..."
```

其余字段（model_path、env、cluster 等）与原 config 完全相同。

---

## 向后兼容

所有现有 config（`kl_beta: 0.0`，无 `ref_policy_path`）：

- `init_worker` 不加载参考权重（`ref_policy_state_dict = None`）
- `run_training` 跳过 reference forward（`ref_logprobs_raw = None`）
- loss 函数内 `_kl_beta > 0` 为 False，KL 项不参与
- **零额外开销，行为完全不变**

---

## logprob 的完整计算链

这个模型是 **Flow Matching（流匹配）** 模型（pi0/pi05），不是 LLM，所以 logprob 不是离散 token 的 softmax，而是**连续高斯分布的对数似然**。

### 第一步：模型输出什么

模型不直接预测动作，而是预测一个**速度场（velocity）** `v_θ(x_t, t)`：

```python
# openpi_action_model.py — sample_mean_var_val
v_t, suffix_out = self.get_velocity(state, x_t, t_input, ...)
```

`x_t` 是去噪过程中第 t 步的带噪动作（从纯高斯噪声 `x_T` 开始，逐步去噪到干净动作 `x_0`）。模型从当前状态和带噪动作预测速度 v_t，然后推算：

```python
x0_pred = x_t - v_t * t_input       # 预测的干净动作
x1_pred = x_t + v_t * (1 - t_input) # 预测的原始噪声
```

### 第二步：怎么变成均值和标准差

以默认的 `flow_sde` 模式为例（有随机噪声，才能算 logprob）：

```python
sigma_i    = sigmas[idx]                                        # 当前步噪声尺度
x_t_mean   = x0_pred * x0_weight + x1_pred * x1_weight         # 下一步期望值
x_t_std    = sqrt(delta) * sigma_i                              # 下一步标准差
```

`flow_ode`（确定性）的 `x_t_std = 0`，分母为零无法算 logprob，**所以 ODE 模式不能用于训练**。

### 第三步：算高斯 log 概率

```python
# get_logprob_norm(sample=x_{t-1}, mu=x_t_mean, sigma=x_t_std)
log_prob = -log(sigma) - 0.5*log(2π) - 0.5*((sample - mu) / sigma)²
         = log N(x_{t-1} | x_t_mean, x_t_std)
```

逐维度计算，shape = `[bsz, action_chunk, action_dim]` = `[bsz, 50, 14]`。

代表：在给定观测和带噪动作 x_t 的条件下，策略采样出 x_{t-1} 的概率的对数。

### 第四步：多个去噪步怎么聚合

去噪过程有 `num_steps`（例如 5）步，每步都能算一个 logprob：

- **`joint_logprob = False`（默认，handover_block 使用）：** 训练时随机抽一个去噪步 index（Monte Carlo 估计），只取那一步的 logprob：
  ```python
  denoise_inds = torch.tensor([random.randint(0, num_steps - 1)] * num_steps)
  log_probs = log_probs[torch.arange(bsz), denoise_inds[:, 0]]
  # shape: [bsz, action_chunk, action_dim]
  ```

- **`joint_logprob = True`：** 所有去噪步都算，然后在步骤维度取均值：
  ```python
  log_probs = log_probs.mean(dim=1)
  ```

### 第五步：training forward 重新算 logprob

rollout 阶段存的是 `chains`（完整去噪轨迹）和 `denoise_inds`（随机选的步骤 index）。training 阶段用 `get_log_prob_value` **重新过一遍模型**拿当前策略的 logprob：

```python
# default_forward (training 阶段)
log_probs, value_t, entropy = self.get_log_prob_value(
    images, img_masks, lang_tokens, lang_masks,
    state, chains, denoise_inds, compute_values
)
log_probs = log_probs[:, :, :action_chunk, :action_env_dim]
log_probs = log_probs.mean(dim=1)   # 对去噪步取均值
# 最终 shape: [bsz, 50, 14]
```

### 第六步：preprocess_loss_inputs 的最后 reshape

`logprobs` 从模型出来是 `[bsz, 50, 14]`，进入 `preprocess_loss_inputs` 按 `logprob_type` 聚合：

```
chunk_level:  [bsz, 50, 14].sum(dim=[1,2]) → [bsz]
              = 一次 action chunk 调用的总 log prob（50 时间步 × 14 关节求和）
```

这个 `[bsz]` 标量就是 PPO ratio 里的 `log π(a|s)`。bsz 是展平后的 action chunk 样本数，不是 episode 数。

### 完整链路

```
观测（图像 + 语言 + 关节状态）
      ↓
  模型预测速度 v_θ(x_t, t)
      ↓
  推算 x_t_mean, x_t_std（flow_sde 分支）
      ↓
  get_logprob_norm: log N(x_{t-1} | x_t_mean, x_t_std)
  shape: [bsz, num_steps, action_chunk, action_dim]
      ↓
  对 num_steps 维度取均值（或选单步）
  shape: [bsz, 50, 14]
      ↓
  preprocess_loss_inputs (chunk_level: .sum(dim=[1,2]))
  shape: [bsz]
      ↓
  PPO ratio = exp(log π_current - log π_old)
```

核心思想：**把 flow matching 的去噪过程当成一个随机策略**，每步去噪 = 从条件高斯采样，logprob = 高斯的对数似然，sum over all dimensions = 整条轨迹的 log π(τ)。

---

## RLinf 官方 KL 约束实现详解（reasoning/LLM 侧）

官方实现分三层：**参考权重管理 → rollout 阶段计算 ref_logprobs → training 阶段加 KL 到 loss**。

### 第一层：参考权重怎么来

官方的参考策略**不是从外部文件加载**，而是在 `init_worker` 时把当前模型（训练起始 checkpoint）快照到 CPU：

```python
# fsdp_actor_worker.py — FSDPActor.init_worker
def init_worker(self):
    self.setup_model_and_optimizer()          # 加载训练起始权重到 GPU
    if (self.kl_beta > 0 or self.reinpp_kl_beta > 0) and self.combine_reference_model:
        self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)
        self.offload_model_buffer = {}
```

`retrieve_model_state_dict_in_cpu` 逐层把 GPU 参数 `.detach().copy()` 到 **pin memory 的 CPU tensor**，避免影响训练 graph，并加速后续的 CPU→GPU 传输。

结论：**参考策略 = 训练开始那一刻的快照**，此后冻结。

### 第二层：ref_logprobs 何时计算

**官方在 rollout 阶段算，不在 training 阶段算。**

Runner 初始化时根据 `kl_beta` 决定是否需要 ref_logprobs：

```python
# reasoning_runner.py
self.compute_ref_logprobs = (
    self.cfg.algorithm.kl_beta > 0
    or self.cfg.algorithm.get("reinpp_kl_beta", 0) > 0
)
```

每轮 rollout 后，由 actor 的 `inference_step` 用 `cpu_weight_swap` 把参考权重临时换入 GPU 做前向：

```python
# fsdp_actor_worker.py — FSDPActor.inference_step
if compute_ref_logprobs:
    with cpu_weight_swap(
        self.model,
        self.ref_policy_state_dict,
        self.offload_model_buffer,
    ):
        ref_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()
```

`cpu_weight_swap` 是一个 context manager：进入时把当前训练权重转移到 CPU、把参考权重加载到 GPU；退出时还原。计算完的 ref_logprobs 存到 `rollout_result.ref_logprobs`，随 batch 传到 training 阶段。

### 第三层：training 阶段怎么加 KL

KL 项在 PPO loss 之外**单独计算**，直接加到总 loss：

```python
# fsdp_actor_worker.py — FSDPActor.run_training
ref_logprobs = m_batch.get("ref_logprobs", None)    # 从 rollout 传来

loss, mbs_metrics_data = policy_loss(...)            # 普通 PPO loss，不含 KL

kl_loss = torch.tensor(0.0, ...)
if self.kl_beta > 0 and ref_logprobs is not None:
    kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
    kl_loss = self.loss_agg_func(kld, loss_mask)
    loss = loss + kl_loss * self.kl_beta             # L_total = L_PPO + β·KL
```

`kl_penalty` 的各模式：

| 模式 | 公式 | 特点 |
|---|---|---|
| `kl` / `k1` | `logprob - ref_logprob` | 最便宜，方差大 |
| `abs` | `\|logprob - ref_logprob\|` | 对称，无方向 |
| `mse` / `k2` | `0.5 * (logprob - ref_logprob)²` | 更平滑 |
| `low_var_kl` / `k3` | `exp(r) - r - 1`，r=ref-log | **方差最低，官方 reasoning 默认** |

---

## 官方实现 vs 本次 embodied 实现的差异

| 维度 | 官方 reasoning 实现 | 本次 embodied 实现 |
|---|---|---|
| **参考策略来源** | 训练起始权重快照（`retrieve_model_state_dict_in_cpu`） | 外部 5-task PPO checkpoint（`torch.load`） |
| **ref_logprobs 计算时机** | **rollout 阶段**（`inference_step`），结果存入 batch | **training 阶段**（每次 mini-batch 前即时计算） |
| **KL 加入位置** | training loop 里，`policy_loss()` 调用之后 | `compute_ppo_actor_loss()` 内部 |
| **数据流** | rollout → `rollout_result.ref_logprobs` → batch → training | training 内即时计算，不经过 rollout | 
| **默认 kl_penalty_type** | `low_var_kl`（k3，方差最低） | `low_var_kl`（k3，与官方对齐） |

核心原因：官方在 rollout 阶段算 ref_logprobs，是因为 reasoning 支持 **disaggregated 流水线**，把 ref forward 塞进 rollout 阶段正好不额外占训练时间。embodied 没有这个流水线，所以在 training 的 update epoch 里每次算更直接。

---

## 验证

快速导入检查（无 GPU）：

```bash
python -c "
from rlinf.algorithms.losses import compute_ppo_actor_loss
import torch
bsz = 4
lp = torch.randn(bsz).requires_grad_(True)
ref_lp = lp.clone().detach()
loss, metrics = compute_ppo_actor_loss(
    logprobs=lp, old_logprobs=lp.detach(),
    advantages=torch.randn(bsz), clip_ratio_low=0.2, clip_ratio_high=0.2,
    ref_logprobs=ref_lp, kl_beta=0.1, kl_penalty_type='kl',
)
print('ref_kl_loss:', metrics['actor/ref_kl_loss'])  # expect ~0.0
"
```

训练验证：`runner.max_steps: 2`，确认 tensorboard 出现 `actor/ref_kl_loss`。

回归验证：用原始 `moe_sft20k_handover_block_ppo100.yaml`（`kl_beta: 0.0`）跑一步，确认 `actor/ref_kl_loss = 0.0`。

---

## 字段拆分（2026-07-19）

Codex review 指出 `kl_beta` 字段被官方 KL 和我们的 ref KL 共用，存在冲突风险。已拆分为两个独立字段：

| 字段 | 归属 | 含义 |
|---|---|---|
| `algorithm.kl_beta` | RLinf 官方 | reasoning/agent 路径的 KL penalty coefficient，不动 |
| `algorithm.ref_kl_beta` | 本次新增 | 5-task reference policy KL 系数，embodied 专用 |

代码改动：
- `losses.py`：读 `ref_kl_beta`；只要 `ref_logprobs` 存在就**始终计算并记录** `actor/ref_kl_loss`，只有 `ref_kl_beta > 0` 时才加入 loss
- `fsdp_actor_worker.py`：加载 ref policy 和触发 ref forward 的条件改为 `ref_policy_path is not None`（不再依赖 `ref_kl_beta`），kwargs 字段名改为 `ref_kl_beta`

这样支持三种模式：

| `ref_policy_path` | `ref_kl_beta` | 效果 |
|---|---|---|
| 未设置 | 任意 | 完全跳过，零开销 |
| 已设置 | `0.0` | 加载并运行 ref forward，**只记录** metric，不影响 loss |
| 已设置 | `> 0` | 加载并运行 ref forward，记录 metric，**加入 loss** |

---

## 实验记录

### Phase 1：KL 量级监控（2026-07-19）

目的：在不施加 KL 约束的情况下，观测 `KL_penalty(π_hb, π_5task)` 的量级，用于确定 `ref_kl_beta` 的合理范围。

**设计：** `ref_kl_beta: 0.0`（纯监控），`ref_policy_path` 指向 5-task step-100，tensorboard 记录 `actor/ref_kl_loss`。

| Job ID | 起点 | Config | 环境数 |
|---|---|---|---|
| `10269577` | SFT moe_sft20k（无 PPO 历史） | `moe_sft20k_handover_block_ppo_klmon_fromsft` | 64 |
| `10269578` | HB PPO global_step_200 | `moe_sft20k_handover_block_ppo_klmon_fromhb200` | 128 |

观测指标：`actor/ref_kl_loss`（low_var_kl，chunk-sum over 50×14=700 dims）

后续：根据两个 job 的 `actor/ref_kl_loss` 曲线范围，确定 `ref_kl_beta`，再启动实际施加 KL 约束的训练（使用 `moe_sft20k_handover_block_ppo_kl.yaml`，当前 `ref_kl_beta: 0.01`，可能需要调整）。
