# 当前代码实现的 Hidden-State Distance 指标说明

## 1. 指标目的

当前 `scripts/analyze_hidden_states.py` 的目标是分析不同 checkpoint 在 π0.5 action expert 内部表征上的差异，用于诊断任务合并后是否出现 task-level representation incompatibility。

对于 π0.5 这类 flow-matching VLA 模型，模型输出动作并不是直接由 prefix hidden state 决定，而是由 action suffix tokens 经过 Action Expert 后得到的 latent 决定。因此代码主要分析：

```python
suffix_out[:, -50:, :]
```

其中：

* `50` 表示 action horizon，即未来 50 个 action token；
* 每个 action token 的 hidden dimension 是 1024；
* 因此 hidden state 的 shape 为：

[
H \in \mathbb{R}^{B \times 50 \times 1024}
]

这里的 (H) 不是最终速度场，而是 `action_out_proj` 之前的 action latent。真正的 flow-matching 速度场为：

[
v_t = \text{action_out_proj}(H)
]

其 shape 为：

[
v_t \in \mathbb{R}^{B \times 50 \times 32}
]

因此，当前代码分析的是 **动作速度场生成前的 action latent representation**。

---

## 2. 当前代码收集的 hidden states

对于每个 checkpoint，代码固定：

* 相同 synthetic observation；
* 相同 noise seed；
* 相同 noisy action (x_t)；
* 相同 denoising time 集合：

[
t \in {0.1,0.3,0.5,0.7,0.9}
]

然后通过 `get_suffix_out` 触发模型 forward，并 hook Action Expert 每一层的 hidden state。

对每个 checkpoint，最终得到：

```python
hidden_states[ckpt][t_val][layer_idx]
```

其中每个 hidden state 的 shape 为：

```python
(B=4, 50, 1024)
```

也就是：

[
H^{(l,t)} \in \mathbb{R}^{4 \times 50 \times 1024}
]

其中：

* (l) 表示 Action Expert 的层编号；
* (t) 表示 denoising time；
* (b) 表示 batch index；
* (k) 表示 action token index；
* hidden vector 维度为 1024。

---

## 3. 当前代码实现的距离公式

对于两个 checkpoint (A) 和 (B)，代码在每一层 (l)、每个 denoising time (t)、每个 batch 样本 (b)、每个 action token (k) 上取出两个 hidden vector：

[
h_A^{(l,t,b,k)} \in \mathbb{R}^{1024}
]

[
h_B^{(l,t,b,k)} \in \mathbb{R}^{1024}
]

然后计算：

[
d_{\mathrm{RHD}}(A,B;l,t,b,k)
=============================

\frac{
\left|
h_A^{(l,t,b,k)}
---------------

h_B^{(l,t,b,k)}
\right|_2
}{
\left|
h_A^{(l,t,b,k)}
\right|_2
+
\left|
h_B^{(l,t,b,k)}
\right|_2
+
\epsilon
}
]

对应代码为：

```python
h_a_k = h_a[:, k, :]   # shape: (B, 1024)
h_b_k = h_b[:, k, :]   # shape: (B, 1024)

diff = np.linalg.norm(h_a_k - h_b_k, axis=-1)
norm_a = np.linalg.norm(h_a_k, axis=-1)
norm_b = np.linalg.norm(h_b_k, axis=-1)

dist = diff / (norm_a + norm_b + 1e-8)
```

因此，代码中 `sim_arr` 实际保存的是一个 **relative hidden distance**，而不是 similarity。

更准确的命名应该是：

```python
dist_arr
```

或：

```python
rhd_arr
```

其中 RHD 表示：

[
\text{Relative Hidden Distance}
]

---

## 4. 全局距离的定义

代码最终将所有层、所有 denoising time、所有 batch 样本、所有 action token 的距离取平均：

[
RHD(A,B)
========

\mathbb{E}*{l,t,b,k}
\left[
d*{\mathrm{RHD}}(A,B;l,t,b,k)
\right]
]

展开写为：

[
RHD(A,B)
========

\frac{1}{L|\mathcal{T}|BK}
\sum_{l=1}^{L}
\sum_{t\in\mathcal{T}}
\sum_{b=1}^{B}
\sum_{k=1}^{K}
\frac{
\left|
h_A^{(l,t,b,k)}
---------------

h_B^{(l,t,b,k)}
\right|_2
}{
\left|
h_A^{(l,t,b,k)}
\right|_2
+
\left|
h_B^{(l,t,b,k)}
\right|_2
+
\epsilon
}
]

其中：

[
L=18,\quad |\mathcal{T}|=5,\quad B=4,\quad K=50
]

所以当前代码报告的全局数字是：

```python
sim_arr.mean()
```

但从语义上讲，它应该理解为：

[
RHD(A,B)
]

即两个 checkpoint 的平均相对 hidden distance。

---

## 5. 当前距离的取值范围

对任意两个非零向量 (a,b)，当前代码的 token-level 距离为：

[
d_{\mathrm{RHD}}(a,b)
=====================

\frac{|a-b|_2}{|a|_2+|b|_2}
]

根据三角不等式：

[
|a-b|_2 \leq |a|_2+|b|_2
]

因此：

[
0 \leq d_{\mathrm{RHD}}(a,b) \leq 1
]

特殊情况包括：

| 情况                    |      距离 |          |              |
| --------------------- | ------: | -------- | ------------ |
| (a=b)                 |       0 |          |              |
| (a=-b)，且 (|a|=|b|)    |       1 |          |              |
| (b=0)，(a\neq 0)       |       1 |          |              |
| (b=\alpha a,\alpha>0) | (\frac{ | 1-\alpha | }{1+\alpha}) |

这说明当前指标越大，表示两个 checkpoint 的 action latent 越不同；越小，表示两个 checkpoint 的 action latent 越接近。

---

## 6. 与论文式 normalized L2 distance 的关系

论文中 Hidden-state Distance Similarity 的核心思想是比较不同模型在同一批输入上的 hidden states 距离，然后将距离通过 max-min 映射成 similarity。

一种常见实现是先对 hidden vector 做 L2 normalize，再计算 L2 distance：

[
d_{\mathrm{norm}}(a,b)
======================

\left|
\frac{a}{|a|_2}
---------------

\frac{b}{|b|_2}
\right|_2
]

令：

[
r=|a|_2,\quad s=|b|_2
]

[
c=\cos\theta
============

\frac{a^\top b}{|a|_2|b|_2}
]

则论文式 normalized L2 distance 为：

[
d_{\mathrm{norm}}(a,b)
======================

\sqrt{2-2c}
]

它只依赖于两个向量的夹角，不依赖于向量范数。

而当前代码距离为：

[
d_{\mathrm{RHD}}(a,b)
=====================

\frac{\sqrt{r^2+s^2-2rsc}}{r+s}
]

平方后：

[
d_{\mathrm{RHD}}^2(a,b)
=======================

\frac{r^2+s^2-2rsc}{(r+s)^2}
]

由于：

[
r^2+s^2-2rsc
============

(r-s)^2+2rs(1-c)
]

又因为：

[
d_{\mathrm{norm}}^2(a,b)
========================

# 2-2c

2(1-c)
]

所以：

[
d_{\mathrm{RHD}}^2(a,b)
=======================

\frac{
(r-s)^2
+
rs,d_{\mathrm{norm}}^2(a,b)
}{
(r+s)^2
}
]

这个等式说明，当前代码距离由两部分组成：

[
d_{\mathrm{RHD}}^2
==================

\underbrace{
\frac{(r-s)^2}{(r+s)^2}
}*{\text{relative norm mismatch}}
+
\underbrace{
\frac{rs}{(r+s)^2}
d*{\mathrm{norm}}^2
}_{\text{direction mismatch}}
]

因此，当前代码实现不是纯方向距离，而是同时捕捉：

1. hidden vector 的方向差异；
2. hidden vector 的相对范数差异。

---

## 7. 等范数情况下的关系

如果两个 hidden vector 的范数相同：

[
r=s
]

则：

[
d_{\mathrm{RHD}}^2(a,b)
=======================

# \frac{r^2d_{\mathrm{norm}}^2(a,b)}{(2r)^2}

\frac{1}{4}d_{\mathrm{norm}}^2(a,b)
]

因此：

[
d_{\mathrm{RHD}}(a,b)
=====================

\frac{1}{2}d_{\mathrm{norm}}(a,b)
]

也就是说，当不同 checkpoint 的 hidden norm 接近时，当前代码距离与 normalized L2 distance 近似只差一个常数因子：

[
d_{\mathrm{RHD}}
\approx
\frac{1}{2}
d_{\mathrm{norm}}
]

因此，在 hidden norm 变化不大的前提下，当前代码指标与论文式 normalized L2 distance 往往会给出相近的排序趋势。

但是这个前提需要通过实验验证，不能仅凭 RMSNorm 或 adaRMSNorm 自动假设成立。

---

## 8. 当前实现与论文 HiddenSim 的差异

当前实现与论文 HiddenSim 思路一致的地方是：

* 都比较不同 checkpoint 在同一输入条件下的 hidden representation；
* 都试图用 hidden representation distance 作为 task compatibility 的代理指标；
* 都可以用于解释 merging collapse 或 merge-induced forgetting。

但当前实现并不是论文 HiddenSim 的严格复现，主要差异包括：

| 方面                        | 论文 HiddenSim 思路             | 当前代码实现                           |
| ------------------------- | --------------------------- | -------------------------------- |
| 比较对象                      | 最后一层 hidden states          | 18 层 Action Expert hidden states |
| 距离形式                      | normalized L2 distance      | (\frac{|a-b|}{|a|+|b|})          |
| 是否保留范数差异                  | 否，主要比较方向                    | 是，包含相对范数失配                       |
| 汇总方式                      | 对 validation datapoints 求平均 | 对 layer、t、batch、token 求平均        |
| 输出语义                      | similarity，越大越相似            | distance，越大越不同                   |
| 是否做 max-min similarity 映射 | 是                           | 当前 raw result 没有做                |

因此，更严谨的表述是：

> 当前代码实现了一个跨层、跨 denoising time、跨 action token 的 norm-sensitive relative hidden distance，用于分析 π0.5 action expert 表征差异。它不是论文 HiddenSim 的逐字复现，但可以作为 task representation incompatibility 的诊断指标。

---

## 9. 如何将当前距离转成 HiddenSim 风格

因为当前代码输出的是距离：

[
RHD(A,B)
]

所以它的方向是：

[
RHD \uparrow
\Rightarrow
\text{representation difference 更大}
]

如果希望使用 HiddenSim 的解释方向，即“越大越相似”，需要将距离反向映射成 similarity。

一种简单方式是：

[
HiddenSim_{\mathrm{RHD}}(A,B)
=============================

1-RHD(A,B)
]

由于：

[
RHD(A,B)\in[0,1]
]

所以：

[
HiddenSim_{\mathrm{RHD}}(A,B)\in[0,1]
]

另一种更接近论文的方式是，对所有 checkpoint pair 的距离集合 (D) 做 max-min 映射：

[
HiddenSim_{\mathrm{RHD}}(A,B)
=============================

\frac{
\max(D)-RHD(A,B)
}{
\max(D)-\min(D)
}
]

其中：

[
D={RHD(i,j)\mid i<j}
]

这样定义后，才可以使用：

[
HiddenSim_{\mathrm{RHD}}\uparrow
\Rightarrow
\text{任务表征更兼容}
]

[
HiddenSim_{\mathrm{RHD}}\downarrow
\Rightarrow
\text{任务表征更不兼容}
]

---

## 10. 与 forgetting 的关系

当前代码指标可以用于解释 merge-induced forgetting，但需要注意方向。

如果直接使用 raw distance，即：

[
RHD(A,B)
]

那么解释链应该是：

[
RHD(A,B)\downarrow
\Rightarrow
\text{两个任务的 action latent 更接近}
\Rightarrow
\text{任务表征更兼容}
\Rightarrow
\text{合并模型更容易同时接近两个 task experts}
\Rightarrow
\text{forgetting 更小}
]

[
RHD(A,B)\uparrow
\Rightarrow
\text{两个任务的 action latent 更远}
\Rightarrow
\text{任务表征更不兼容}
\Rightarrow
\text{合并模型更容易偏向某一个 expert}
\Rightarrow
\text{另一个任务更容易被牺牲}
\Rightarrow
\text{forgetting 更严重}
]

如果将 RHD 转成 similarity：

[
HiddenSim_{\mathrm{RHD}}(A,B)
=============================

\frac{
\max(D)-RHD(A,B)
}{
\max(D)-\min(D)
}
]

那么可以写成论文风格的解释链：

[
HiddenSim_{\mathrm{RHD}}\uparrow
\Rightarrow
\text{任务表征更兼容}
\Rightarrow
\text{合并时更容易保留各任务能力}
\Rightarrow
\text{forgetting 更小}
]

[
HiddenSim_{\mathrm{RHD}}\downarrow
\Rightarrow
\text{任务表征不兼容}
\Rightarrow
\text{合并时更容易牺牲某些任务}
\Rightarrow
\text{forgetting 更严重}
]

---

## 11. 为什么该指标可以解释 merge-induced forgetting

设两个任务专家分别为：

[
\theta_A,\theta_B
]

合并模型为：

[
\theta_M
]

在同一个 observation、同一个 noisy action (x_t)、同一个 denoising time (t) 下，两个专家的 action latent 分别为：

[
h_A(o,x_t,t)
]

[
h_B(o,x_t,t)
]

如果：

[
RHD(A,B)
]

较小，说明：

[
h_A(o,x_t,t)\approx h_B(o,x_t,t)
]

此时存在一个 merged representation 同时接近两个专家：

[
h_M(o,x_t,t)\approx h_A(o,x_t,t)\approx h_B(o,x_t,t)
]

因此，合并模型更容易同时保留两个任务的动作生成能力。

反之，如果：

[
RHD(A,B)
]

较大，说明两个任务专家在 action latent space 中距离较远。此时合并模型很难同时接近两个专家：

[
h_M\approx h_A
\Rightarrow
h_M \not\approx h_B
]

或者：

[
h_M\approx h_B
\Rightarrow
h_M \not\approx h_A
]

这会导致 merged model 更像其中一个 task expert，而远离另一个 task expert，从而表现为某个任务性能保留较好，另一个任务发生 forgetting。

---

## 12. 当前指标的适用解释

当前代码中的 RHD 指标适合解释以下问题：

1. 不同 PPO checkpoint 在 action latent space 中是否分化；
2. merged checkpoint 更接近哪个 task expert；
3. 表征差异是在哪些 layer 开始放大的；
4. 表征差异是否集中在某些 denoising time；
5. 表征差异是否集中在某些 action horizon token；
6. merge 后某个任务性能下降是否与 action latent 偏离有关。

因此，它特别适合用于分析：

[
\theta_{\text{SFT}},
\theta_{\text{PPO-CA}},
\theta_{\text{PPO-ADJ}},
\theta_{\text{merged}}
]

之间的 representation distance。

如果发现：

[
RHD(\theta_{\text{merged}},\theta_{\text{PPO-CA}})
<
RHD(\theta_{\text{merged}},\theta_{\text{PPO-ADJ}})
]

并且 merged model 在 CA 上表现更好、在 ADJ 上表现更差，那么可以解释为：

> merged model 的 action latent 更接近 CA expert，因此更好地保留了 CA 能力；同时它远离 ADJ expert，因此 ADJ 发生了更明显的 forgetting。

---

## 13. 建议报告方式

为了避免术语混淆，建议在实验结果中同时报告两个名称：

### 13.1 Raw distance

[
RHD(A,B)
========

\mathbb{E}_{l,t,b,k}
\left[
\frac{
|h_A^{(l,t,b,k)}-h_B^{(l,t,b,k)}|_2
}{
|h_A^{(l,t,b,k)}|_2+|h_B^{(l,t,b,k)}|_2+\epsilon
}
\right]
]

含义：

[
RHD \uparrow
\Rightarrow
\text{hidden representation 更不同}
]

### 13.2 Similarity version

[
HiddenSim_{\mathrm{RHD}}(A,B)
=============================

\frac{
\max(D)-RHD(A,B)
}{
\max(D)-\min(D)
}
]

含义：

[
HiddenSim_{\mathrm{RHD}}\uparrow
\Rightarrow
\text{hidden representation 更相似}
]

---

## 14. 最终总结

当前代码实现的不是论文 HiddenSim 的严格复现，而是一个更适合 π0.5 的 action-latent diagnostic metric：

[
\boxed{
RHD(A,B)
========

\mathbb{E}_{l,t,b,k}
\left[
\frac{
|h_A^{(l,t,b,k)}-h_B^{(l,t,b,k)}|_2
}{
|h_A^{(l,t,b,k)}|_2+|h_B^{(l,t,b,k)}|_2+\epsilon
}
\right]
}
]

它衡量的是：

[
\boxed{
\text{direction mismatch}
+
\text{relative norm mismatch}
}
]

因此它可以作为 task representation incompatibility 的代理指标。

如果直接使用 RHD，则解释方向是：

[
\boxed{
RHD\downarrow
\Rightarrow
\text{任务表征更兼容}
\Rightarrow
\text{forgetting 更小}
}
]

[
\boxed{
RHD\uparrow
\Rightarrow
\text{任务表征更不兼容}
\Rightarrow
\text{forgetting 更严重}
}
]

如果将其反向映射为 HiddenSim，则可以使用：

[
\boxed{
HiddenSim_{\mathrm{RHD}}\uparrow
\Rightarrow
\text{任务表征更兼容}
\Rightarrow
\text{合并时更容易保留各任务能力}
\Rightarrow
\text{forgetting 更小}
}
]

[
\boxed{
HiddenSim_{\mathrm{RHD}}\downarrow
\Rightarrow
\text{任务表征不兼容}
\Rightarrow
\text{合并时更容易牺牲某些任务}
\Rightarrow
\text{forgetting 更严重}
}
]
