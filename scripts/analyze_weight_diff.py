#!/usr/bin/env python3
"""SFT 权重 vs RL 权重差异分析脚本（实验一 + 实验二）。

实验一：逐层参数变化量分析
    - τ_SFT = ca_sft15k - rf20sft
    - τ_PPO = merged_a07_ppo_adj_step50 - merged_a07
    - 指标：rel_change（Frobenius 范数比）、cos_sim（余弦相似度）

实验二：任务向量干扰分析
    - cos(τ_SFT, τ_PPO) 按层分组
    - < 0 → 方向冲突 → 合并崩溃预测因子

用法：
    python scripts/analyze_weight_diff.py \\
        --rf20sft   <path>/random_first20_pi05_sft/full_weights.pt \\
        --ca-sft15k <path>/global_step_15000/.../full_weights.pt \\
        --merged-a07 <path>/merged_weights/ppo_adj200_plus_ca100_tv_a07/full_weights.pt \\
        --ppo-step50 <path>/global_step_50/.../full_weights.pt \\
        --output-dir results/weight_diff/
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import re
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Pi0.5 PPO 实际可训练参数白名单（208/812 keys）
PI05_PPO_TRAINABLE_PREFIXES: tuple[str, ...] = (
    "paligemma_with_expert.gemma_expert.model.layers.",
    "paligemma_with_expert.gemma_expert.model.norm.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)

# 模块类型关键词映射（顺序从长到短，避免短串误匹配）
MODULE_KEYWORDS: list[tuple[str, str]] = [
    ("input_layernorm", "input_layernorm"),
    ("post_feedforward_layernorm", "post_feedforward_layernorm"),
    ("post_attention_layernorm", "post_attention_layernorm"),
    ("q_proj", "q_proj"),
    ("k_proj", "k_proj"),
    ("v_proj", "v_proj"),
    ("o_proj", "o_proj"),
    ("gate_proj", "gate_proj"),
    ("up_proj", "up_proj"),
    ("down_proj", "down_proj"),
    ("action_in_proj", "action_in_proj"),
    ("action_out_proj", "action_out_proj"),
    ("time_mlp_in", "time_mlp_in"),
    ("time_mlp_out", "time_mlp_out"),
    ("norm", "norm"),
]

# 绘图时使用固定颜色映射
MODULE_COLORS: dict[str, str] = {
    "q_proj": "#1f77b4",
    "k_proj": "#aec7e8",
    "v_proj": "#2ca02c",
    "o_proj": "#98df8a",
    "gate_proj": "#d62728",
    "up_proj": "#ff9896",
    "down_proj": "#9467bd",
    "input_layernorm": "#8c564b",
    "post_feedforward_layernorm": "#e377c2",
    "post_attention_layernorm": "#7f7f7f",
    "action_in_proj": "#bcbd22",
    "action_out_proj": "#17becf",
    "time_mlp_in": "#ff7f0e",
    "time_mlp_out": "#ffbb78",
    "norm": "#c7c7c7",
    "other": "#000000",
}


def load_pt(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    sd = torch.load(str(path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
        sd = sd["state_dict"]
    logger.info(f"  已加载 {len(sd)} 个张量：{path}")
    return sd


def classify_key(key: str) -> tuple[Optional[int], str]:
    """提取 (layer_idx, module_type)。非 Expert 层返回 (None, module_type)。"""
    m = re.search(r"\.layers\.(\d+)\.", key)
    layer_idx = int(m.group(1)) if m else None
    for substr, label in MODULE_KEYWORDS:
        if substr in key:
            return layer_idx, label
    return layer_idx, "other"


def compute_task_vector_stats(
    base_sd: dict[str, torch.Tensor],
    ft_sd: dict[str, torch.Tensor],
    prefix_filter: tuple[str, ...] = PI05_PPO_TRAINABLE_PREFIXES,
) -> list[dict]:
    """计算每个可训练 key 的变化量统计。"""
    rows = []
    for key in base_sd:
        if not any(key.startswith(p) for p in prefix_filter):
            continue
        if key not in ft_sd:
            logger.warning(f"  跳过（ft 中缺失）：{key}")
            continue
        base = base_sd[key].to(torch.float32)
        ft = ft_sd[key].to(torch.float32)
        delta = ft - base

        base_norm = base.norm().item()
        delta_norm = delta.norm().item()
        rel_change = delta_norm / (base_norm + 1e-12)
        cos_sim = F.cosine_similarity(base.flatten().unsqueeze(0),
                                       ft.flatten().unsqueeze(0)).item()

        layer_idx, module_type = classify_key(key)
        rows.append({
            "key": key,
            "layer_idx": layer_idx,
            "module_type": module_type,
            "base_norm": base_norm,
            "ft_norm": ft.norm().item(),
            "delta_norm": delta_norm,
            "rel_change": rel_change,
            "cos_sim": cos_sim,
        })
    return rows


def compute_interference(
    base_a: dict[str, torch.Tensor],
    ft_a: dict[str, torch.Tensor],
    base_b: dict[str, torch.Tensor],
    ft_b: dict[str, torch.Tensor],
    prefix_filter: tuple[str, ...] = PI05_PPO_TRAINABLE_PREFIXES,
) -> list[dict]:
    """计算两个任务向量之间的余弦干扰（实验二）。"""
    rows = []
    common_keys = [
        k for k in base_a
        if any(k.startswith(p) for p in prefix_filter)
        and k in ft_a and k in base_b and k in ft_b
    ]
    for key in common_keys:
        tau_a = (ft_a[key].to(torch.float32) - base_a[key].to(torch.float32)).flatten()
        tau_b = (ft_b[key].to(torch.float32) - base_b[key].to(torch.float32)).flatten()
        cos_ab = F.cosine_similarity(tau_a.unsqueeze(0), tau_b.unsqueeze(0)).item()
        layer_idx, module_type = classify_key(key)
        rows.append({
            "key": key,
            "layer_idx": layer_idx,
            "module_type": module_type,
            "cos_tau_A_tau_B": cos_ab,
            "norm_tau_A": tau_a.norm().item(),
            "norm_tau_B": tau_b.norm().item(),
        })
    return rows


def plot_layer_curves(df: pd.DataFrame, col_a: str, col_b: str,
                      label_a: str, label_b: str,
                      ylabel: str, title: str, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax, col, label in [(axes[0], col_a, label_a), (axes[1], col_b, label_b)]:
        for mtype, grp in df.groupby("module_type"):
            g = grp.dropna(subset=["layer_idx"]).sort_values("layer_idx")
            if g.empty:
                continue
            color = MODULE_COLORS.get(mtype, "#000000")
            ax.plot(g["layer_idx"], g[col], marker="o", markersize=3,
                    label=mtype, color=color)
        ax.set_xlabel("Action Expert Layer Index")
        ax.set_ylabel(ylabel)
        ax.set_title(label)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"  saved: {out_path}")


def plot_interference_heatmap(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """绘制干扰热力图：18 层 × module_type。"""
    pivot = df.dropna(subset=["layer_idx"]).pivot_table(
        index="module_type", columns="layer_idx", values="cos_tau_A_tau_B",
        aggfunc="mean"
    )
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="cos(tau_SFT, tau_PPO)")
    ax.set_title("Task Vector Interference Heatmap  (blue=aligned, red=conflicting)")
    ax.set_xlabel("Layer Index")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"  已保存：{out_path}")


def plot_interference_by_layer(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    """绘制干扰折线图（按 module_type 分组）。"""
    fig, ax = plt.subplots(figsize=(12, 5))
    for mtype, grp in df.groupby("module_type"):
        g = grp.dropna(subset=["layer_idx"]).sort_values("layer_idx")
        if g.empty:
            continue
        color = MODULE_COLORS.get(mtype, "#000000")
        ax.plot(g["layer_idx"], g["cos_tau_A_tau_B"], marker="o",
                markersize=4, label=mtype, color=color)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.05, color="gray", linewidth=0.6, linestyle=":", label="Libero baseline ~0.05")
    ax.set_xlabel("Action Expert Layer Index")
    ax.set_ylabel("cos(tau_SFT, tau_PPO)")
    ax.set_title("Task Vector Interference: SFT vs PPO  (< 0 = conflicting directions)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"  已保存：{out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="SFT vs RL 权重差异分析（实验一 + 二）")
    ap.add_argument("--rf20sft", required=True, help="rf20sft base full_weights.pt")
    ap.add_argument("--ca-sft15k", required=True, dest="ca_sft15k",
                    help="ca_sft15k full_weights.pt（SFT 专化检查点）")
    ap.add_argument("--merged-a07", required=True, dest="merged_a07",
                    help="merged_a07 full_weights.pt（PPO 出发点）")
    ap.add_argument("--ppo-step50", required=True, dest="ppo_step50",
                    help="merged_a07_ppo_adj_step50 full_weights.pt")
    ap.add_argument("--output-dir", default="results/weight_diff", dest="output_dir")
    ap.add_argument("--include-all", action="store_true",
                    help="分析所有参数（不限于可训练前缀）")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix_filter = None if args.include_all else PI05_PPO_TRAINABLE_PREFIXES

    # ── 加载权重 ──────────────────────────────────────────────
    logger.info("=== 加载权重文件 ===")
    sd_rf20 = load_pt(args.rf20sft)
    sd_ca15k = load_pt(args.ca_sft15k)
    sd_merged = load_pt(args.merged_a07)
    sd_ppo50 = load_pt(args.ppo_step50)

    # ── 实验一：逐层变化量分析 ────────────────────────────────
    logger.info("=== 实验一：逐层参数变化量分析 ===")
    logger.info("  计算 τ_SFT = ca_sft15k - rf20sft ...")
    rows_sft = compute_task_vector_stats(sd_rf20, sd_ca15k, prefix_filter or PI05_PPO_TRAINABLE_PREFIXES)
    logger.info("  计算 τ_PPO = merged_a07_ppo_adj_step50 - merged_a07 ...")
    rows_ppo = compute_task_vector_stats(sd_merged, sd_ppo50, prefix_filter or PI05_PPO_TRAINABLE_PREFIXES)

    df_sft = pd.DataFrame(rows_sft).add_suffix("_SFT").rename(columns={"key_SFT": "key"})
    df_ppo = pd.DataFrame(rows_ppo).add_suffix("_PPO").rename(columns={"key_PPO": "key"})
    df_merged = df_sft.merge(df_ppo, on="key", suffixes=("", ""))
    # 修复：去重 layer_idx / module_type 列
    for col in ["layer_idx", "module_type"]:
        col_sft = f"{col}_SFT"
        col_ppo = f"{col}_PPO"
        if col_sft in df_merged.columns:
            df_merged[col] = df_merged[col_sft]
            df_merged.drop(columns=[c for c in [col_sft, col_ppo] if c in df_merged.columns],
                           inplace=True)

    per_key_path = out_dir / "per_key_stats.csv"
    df_merged.to_csv(per_key_path, index=False)
    logger.info(f"  已保存：{per_key_path}（{len(df_merged)} 行）")

    # 按 (layer_idx, module_type) 分组
    df_layer = (
        df_merged.dropna(subset=["layer_idx"])
        .groupby(["layer_idx", "module_type"])[["rel_change_SFT", "cos_sim_SFT", "rel_change_PPO", "cos_sim_PPO"]]
        .mean()
        .reset_index()
    )
    layer_path = out_dir / "layerwise_summary.csv"
    df_layer.to_csv(layer_path, index=False)
    logger.info(f"  已保存：{layer_path}")

    # 绘制 rel_change 折线图
    plot_layer_curves(
        df_layer,
        col_a="rel_change_SFT", col_b="rel_change_PPO",
        label_a="τ_SFT（ca_sft15k - rf20sft）",
        label_b="τ_PPO（ppo_adj_step50 - merged_a07）",
        ylabel="相对变化量 ||Δ||_F / ||θ_base||_F",
        title="实验一：逐层参数变化量对比",
        out_path=out_dir / "layer_rel_change_curves.png",
    )
    plot_layer_curves(
        df_layer,
        col_a="cos_sim_SFT", col_b="cos_sim_PPO",
        label_a="τ_SFT（ca_sft15k - rf20sft）",
        label_b="τ_PPO（ppo_adj_step50 - merged_a07）",
        ylabel="余弦相似度 cos(θ_base, θ_ft)",
        title="实验一：逐层余弦相似度对比",
        out_path=out_dir / "layer_cosine_curves.png",
    )

    # ── 实验二：任务向量干扰分析 ──────────────────────────────
    logger.info("=== 实验二：任务向量干扰分析 ===")
    rows_intf = compute_interference(
        sd_rf20, sd_ca15k,
        sd_merged, sd_ppo50,
        prefix_filter or PI05_PPO_TRAINABLE_PREFIXES,
    )
    df_intf = pd.DataFrame(rows_intf)

    intf_key_path = out_dir / "interference_per_key.csv"
    df_intf.to_csv(intf_key_path, index=False)
    logger.info(f"  已保存：{intf_key_path}（{len(df_intf)} 行）")

    df_intf_layer = (
        df_intf.dropna(subset=["layer_idx"])
        .groupby(["layer_idx", "module_type"])[["cos_tau_A_tau_B"]]
        .mean()
        .reset_index()
    )
    intf_layer_path = out_dir / "interference_layerwise.csv"
    df_intf_layer.to_csv(intf_layer_path, index=False)
    logger.info(f"  已保存：{intf_layer_path}")

    # 汇总统计
    global_mean = df_intf["cos_tau_A_tau_B"].mean()
    global_neg = (df_intf["cos_tau_A_tau_B"] < 0).mean() * 100
    logger.info(f"  全局平均干扰余弦：{global_mean:.4f}（负值比例：{global_neg:.1f}%）")
    logger.info(f"  参考：Libero 任务间基线 ~0.05（近似正交）")

    # 按 module_type 汇总
    module_summary = df_intf.groupby("module_type")["cos_tau_A_tau_B"].agg(
        ["mean", "min", "max", "count"]
    ).sort_values("mean")
    logger.info(f"\n模块类型汇总：\n{module_summary.to_string()}")
    module_summary.to_csv(out_dir / "interference_by_module.csv")

    plot_interference_heatmap(df_intf_layer, out_dir / "interference_heatmap.png")
    plot_interference_by_layer(df_intf_layer, out_dir / "interference_by_layer.png")

    logger.info(f"\n=== 完成 ===\n输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
