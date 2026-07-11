#!/usr/bin/env python3
"""计算 PPO-CA 和 PPO-adj 任务向量在 Action Expert 各层的逐层余弦相似度。

τ_CA  = θ_ppo_ca  - θ_sft20k
τ_adj = θ_ppo_adj - θ_sft20k

逐层：将第 l 层所有参数差值展平拼接成一个向量，计算两向量间的余弦相似度。

用法（CPU，约 2 分钟）：
    python scripts/analyze_ppo_task_interference.py \
        --sft20k   <path>/random_first20_pi05_sft/full_weights.pt \
        --ppo-ca   <path>/global_step_100/.../full_weights.pt \
        --ppo-adj  <path>/global_step_200/.../full_weights.pt \
        --output-dir results/ppo_interference/
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PI05_ACTION_EXPERT_PREFIXES: tuple[str, ...] = (
    "paligemma_with_expert.gemma_expert.model.layers.",
    "paligemma_with_expert.gemma_expert.model.norm.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)

NUM_LAYERS = 18


def load_pt(path: str) -> dict[str, torch.Tensor]:
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
        sd = sd["state_dict"]
    logger.info(f"  loaded {len(sd)} tensors: {path}")
    return sd


def get_layer_idx(key: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def compute_layerwise_cosine(
    sd_base: dict[str, torch.Tensor],
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
) -> dict:
    """逐层拼接任务向量，计算余弦相似度。

    返回:
        {
          "layer_cos": np.ndarray shape (18,),   # 18 层 Expert 层
          "non_layer_cos": float,                 # norm/action_in_out/time_mlp 合并
          "layer_norm_a": np.ndarray (18,),       # 每层 τ_CA 的 L2 范数
          "layer_norm_b": np.ndarray (18,),       # 每层 τ_adj 的 L2 范数
        }
    """
    # 筛出 Action Expert keys
    ae_keys = [
        k for k in sd_base
        if any(k.startswith(p) for p in PI05_ACTION_EXPERT_PREFIXES)
        and k in sd_a and k in sd_b
    ]
    logger.info(f"  Action Expert keys: {len(ae_keys)}")

    # 按层收集差值向量
    layer_vecs_a: dict[int | str, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}
    layer_vecs_b: dict[int | str, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}
    non_layer_vecs_a: list[torch.Tensor] = []
    non_layer_vecs_b: list[torch.Tensor] = []

    for k in ae_keys:
        base = sd_base[k].to(torch.float32)
        ta = (sd_a[k].to(torch.float32) - base).flatten()
        tb = (sd_b[k].to(torch.float32) - base).flatten()
        li = get_layer_idx(k)
        if li is not None and 0 <= li < NUM_LAYERS:
            layer_vecs_a[li].append(ta)
            layer_vecs_b[li].append(tb)
        else:
            non_layer_vecs_a.append(ta)
            non_layer_vecs_b.append(tb)

    # 逐层计算余弦
    layer_cos = np.zeros(NUM_LAYERS)
    layer_norm_a = np.zeros(NUM_LAYERS)
    layer_norm_b = np.zeros(NUM_LAYERS)
    for li in range(NUM_LAYERS):
        if not layer_vecs_a[li]:
            continue
        va = torch.cat(layer_vecs_a[li])
        vb = torch.cat(layer_vecs_b[li])
        layer_cos[li] = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        layer_norm_a[li] = va.norm().item()
        layer_norm_b[li] = vb.norm().item()

    # 非层 key（norm / action_in_out / time_mlp）合并
    non_layer_cos = float("nan")
    if non_layer_vecs_a:
        va = torch.cat(non_layer_vecs_a)
        vb = torch.cat(non_layer_vecs_b)
        non_layer_cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
        logger.info(f"  非层 key 余弦相似度 (norm/proj/mlp): {non_layer_cos:.4f}")

    return {
        "layer_cos": layer_cos,
        "non_layer_cos": non_layer_cos,
        "layer_norm_a": layer_norm_a,
        "layer_norm_b": layer_norm_b,
    }


def plot_results(result: dict, out_dir: pathlib.Path) -> None:
    layer_cos = result["layer_cos"]
    layer_norm_a = result["layer_norm_a"]
    layer_norm_b = result["layer_norm_b"]
    layers = np.arange(NUM_LAYERS)

    # 图1：逐层余弦相似度折线 + 水平参考线
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, layer_cos, marker="o", markersize=5, color="#1f77b4",
            linewidth=1.8, label="cos(τ_CA, τ_adj)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", label="cos = 0")
    ax.axhline(result["non_layer_cos"], color="#ff7f0e", linewidth=1.2,
               linestyle=":", label=f"non-layer keys: {result['non_layer_cos']:.3f}")
    ax.set_xlabel("Action Expert Layer Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Per-layer Cosine Similarity: τ_CA vs τ_adj\n(τ = PPO_checkpoint − sft20k)")
    ax.set_xticks(layers)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "layerwise_cosine_CA_vs_adj.png", dpi=150)
    plt.close()
    logger.info(f"  saved: {out_dir / 'layerwise_cosine_CA_vs_adj.png'}")

    # 图2：双轴 — 余弦相似度（左） + 任务向量范数（右）
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(layers, layer_cos, marker="o", markersize=5, color="#1f77b4",
             linewidth=1.8, label="cos(τ_CA, τ_adj)")
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.bar(layers - 0.2, layer_norm_a, width=0.35, alpha=0.4, color="#d62728", label="‖τ_CA‖")
    ax2.bar(layers + 0.2, layer_norm_b, width=0.35, alpha=0.4, color="#2ca02c", label="‖τ_adj‖")
    ax1.set_xlabel("Action Expert Layer Index")
    ax1.set_ylabel("Cosine Similarity", color="#1f77b4")
    ax2.set_ylabel("Task Vector L2 Norm")
    ax1.set_title("Per-layer Cosine Similarity & Task Vector Norms")
    ax1.set_xticks(layers)
    ax1.set_ylim(-1.05, 1.05)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "layerwise_cosine_with_norms.png", dpi=150)
    plt.close()
    logger.info(f"  saved: {out_dir / 'layerwise_cosine_with_norms.png'}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--sft20k",  required=True, help="Base SFT checkpoint (rf20sft)")
    ap.add_argument("--ppo-ca",  required=True, dest="ppo_ca",
                    help="PPO click_alarmclock checkpoint")
    ap.add_argument("--ppo-adj", required=True, dest="ppo_adj",
                    help="PPO adjust_bottle checkpoint")
    ap.add_argument("--output-dir", default="results/ppo_interference", dest="output_dir")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== 加载权重 ===")
    sd_base = load_pt(args.sft20k)
    sd_ca   = load_pt(args.ppo_ca)
    sd_adj  = load_pt(args.ppo_adj)

    logger.info("=== 计算逐层余弦相似度 ===")
    result = compute_layerwise_cosine(sd_base, sd_ca, sd_adj)

    logger.info("\n===== 结果汇总 =====")
    logger.info(f"{'层':>4}  {'cos(τ_CA, τ_adj)':>18}  {'‖τ_CA‖':>10}  {'‖τ_adj‖':>10}")
    for li in range(NUM_LAYERS):
        logger.info(
            f"{li:>4}  {result['layer_cos'][li]:>18.4f}"
            f"  {result['layer_norm_a'][li]:>10.4f}"
            f"  {result['layer_norm_b'][li]:>10.4f}"
        )
    logger.info(f"非层 key 余弦: {result['non_layer_cos']:.4f}")
    logger.info(f"全层平均余弦: {result['layer_cos'].mean():.4f}")
    logger.info(f"全层最小余弦: {result['layer_cos'].min():.4f}  (层 {result['layer_cos'].argmin()})")
    logger.info(f"全层最大余弦: {result['layer_cos'].max():.4f}  (层 {result['layer_cos'].argmax()})")

    logger.info("=== 绘图 ===")
    plot_results(result, out_dir)

    # 保存数值
    np.save(out_dir / "layerwise_cosine.npy", result["layer_cos"])
    logger.info(f"  saved: {out_dir / 'layerwise_cosine.npy'}")


if __name__ == "__main__":
    main()
