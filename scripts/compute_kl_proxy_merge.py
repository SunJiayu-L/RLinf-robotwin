#!/usr/bin/env python3
"""计算各权重相对于 SFT-20k base 的 KL 代理指标，并与 HiddenSim 对比（真实观测修正版）。

本文件取代同名旧版本。旧版本有两个方法论问题（与 euler 分析文档记录的问题同类）：
  1. 用跨任务共享的随机像素/关节角 + 写死的 "beat block hammer" 指令做观测。
  2. `vel_div` 用同一份固定随机噪声喂给所有 t（0.1~0.9），这正是
     `analyze_hidden_states_moe.py` 里已被认定为 OOD、不可信的"旧方法"，只是换了个名字。
本次修正：
  · 观测改用 moe-demo-clean 真实录制数据（复用 plot_euler_trajectory_heatmaps.py
    里的 load_real_obs_by_name），每个任务用它自己的真实观测和真实指令。
  · vel_div 改成在真正的 Euler 轨迹上采样 x_t（而不是固定噪声），并同时报告
    self-trajectory（闭环，各走各的轨迹）和 same-trajectory（开环，固定用 base
    的 x_t 序列，隔离参数效应）两种口径。
  · action_div 语义上只有 self-trajectory 有意义（开环下 x_0 就等于 base 自己的终点，
    没有信息量），故只报告 self-trajectory。
  · merged 权重分别在 A、B 各自的真实观测上评测，不再用单一（且错误）观测代表两任务。
  · HiddenSim 对比改用本仓库已验证的 `results/euler_trajectory_heatmaps_v2/` 数据
    （同一批真实观测算出来的），不再依赖已不存在的 `results/pairwise_merge_analysis/`。

两个 KL 代理指标：
  1. VelDiv(w, base) = 在 Euler 轨迹的每个 t 上，||v_base - v_w||² 的归一化 L2 差，
     对所有 t、batch 汇总成一个标量。
  2. ActionDiv(w, base) = 固定同一噪声种子跑完整 Euler ODE 后最终 action 的差
     （self-trajectory，即真实部署时会产生的动作）。

用法：
    python scripts/compute_kl_proxy_merge.py \
        --output-dir results/kl_proxy_analysis_v2/ \
        --device cuda --batch-size 4
"""
from __future__ import annotations

import argparse
import csv
import logging
import pathlib
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)

REPO_PATH = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin"
sys.path.insert(0, REPO_PATH)
sys.path.insert(0, str(pathlib.Path(REPO_PATH) / "scripts"))

from plot_euler_trajectory_heatmaps import (  # noqa: E402
    ACTION_DIM_INTERNAL,
    ACTION_HORIZON,
    EULER_STEPS,
    EVAL_RESULTS,
    MERGED_WEIGHTS,
    MOE_SFT20K,
    PAIRS,
    PPO_WEIGHTS,
    SINGLE_TASK_BASELINES,
    TASK_NAMES,
    TASK_ORDER,
    build_model,
    build_prefix_cache,
    load_real_obs_by_name,
    load_weights,
)

ALPHA_MERGE = 0.7   # 融合系数（用于 KL_excess 计算，与 task_arithmetic 一致）
HIDDENSIM_DIR = pathlib.Path(REPO_PATH) / "results" / "euler_trajectory_heatmaps_v2"


# ── KL 代理指标计算 ───────────────────────────────────────────────────────────

def _norm_l2_div(a: np.ndarray, b: np.ndarray) -> float:
    """归一化 L2 散度：||a - b|| / (||a|| + ||b|| + ε)，对全部元素汇总。"""
    diff = np.linalg.norm((a - b).reshape(-1))
    norm_a = np.linalg.norm(a.reshape(-1))
    norm_b = np.linalg.norm(b.reshape(-1))
    return float(diff / (norm_a + norm_b + 1e-8))


def _get_velocity(model, state, x_t, t_tensor, prefix_pad_masks, past_key_values):
    suffix_out = model.get_suffix_out(state, prefix_pad_masks, past_key_values, x_t, t_tensor)
    proj_dtype = model.action_out_proj.weight.dtype
    v_t = model.action_out_proj(suffix_out.to(proj_dtype))
    return v_t.float()


@torch.no_grad()
def run_euler_and_collect_velocities(model, pfx_pads, pkv, state, x_1, device,
                                      num_steps: int = EULER_STEPS):
    """真实 Euler 轨迹积分（self-trajectory），同时记录每步的速度场和最终动作。

    Returns:
        vels: (num_steps, B, ACTION_HORIZON, ACTION_DIM_INTERNAL) float32
        final_action: (B, ACTION_HORIZON, ACTION_DIM_INTERNAL) float32，积分终点 x_0
        x_traj: list[Tensor]，每步积分前实际使用的 x_t（供 same-trajectory 复用）
        t_vals: list[float]
    """
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    x = x_1.clone().to(dtype).to(device)
    dt = -1.0 / num_steps

    vels, x_traj, t_vals = [], [], []
    for i in range(num_steps):
        t_val = 1.0 + i * dt
        t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
        x_traj.append(x.detach().clone())
        v_t = _get_velocity(model, state, x, t_tensor, pfx_pads, pkv)
        vels.append(v_t.cpu().numpy())
        t_vals.append(round(t_val, 1))
        x = x + dt * v_t.to(dtype)

    return np.stack(vels, axis=0), x.float().cpu().numpy(), x_traj, t_vals


@torch.no_grad()
def probe_velocities_fixed_trajectory(model, pfx_pads, pkv, state,
                                       x_traj: list[torch.Tensor], t_vals: list[float],
                                       device: str) -> np.ndarray:
    """same-trajectory：在外部给定的固定 x_t 序列上算速度场，不做自身积分。"""
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    vels = []
    for x_t, t_val in zip(x_traj, t_vals):
        t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
        v_t = _get_velocity(model, state, x_t.to(dtype).to(device), t_tensor, pfx_pads, pkv)
        vels.append(v_t.cpu().numpy())
    return np.stack(vels, axis=0)


def compute_proxies_on_obs(model, weight_path: str, obs: dict, x_1, device,
                            base_vels_self, base_action, base_x_traj, base_t_vals) -> dict:
    """加载 weight_path，在给定真实观测 obs 上算 vel_div(self/same) 和 action_div(self)。

    调用前 base_vels_self / base_action / base_x_traj / base_t_vals 必须是用同一个 obs
    算出来的 base 结果（不同任务的 obs 不同，因此 base 也要按 obs 重算）。
    """
    load_weights(model, weight_path)
    model.eval()
    pfx, pkv, st = build_prefix_cache(model, obs, device)

    w_vels_self, w_action, _, _ = run_euler_and_collect_velocities(model, pfx, pkv, st, x_1, device)
    w_vels_same = probe_velocities_fixed_trajectory(model, pfx, pkv, st, base_x_traj, base_t_vals, device)

    return {
        "vel_div_self": _norm_l2_div(base_vels_self, w_vels_self),
        "vel_div_same": _norm_l2_div(base_vels_self, w_vels_same),
        "action_div": _norm_l2_div(base_action, w_action),
    }


def load_hiddensim_baseline() -> dict:
    """从已验证的 euler_trajectory_heatmaps_v2 读取 HiddenSim（self-trajectory max），
    作为与本脚本 KL 代理指标同一批真实观测下的对比基准。"""
    single = {}
    for task in TASK_ORDER:
        p = HIDDENSIM_DIR / f"euler_hs_{task}_vs_base.npz"
        if p.exists():
            single[task] = float(np.load(p)["self"].max())

    merged = {}
    for ta, tb in PAIRS:
        p = HIDDENSIM_DIR / f"heatmap_{ta}_{tb}.npz"
        if p.exists():
            d = np.load(p)
            merged[(ta, tb)] = {
                "on_ta": float(d["self_on_ta"].max()),
                "on_tb": float(d["self_on_tb"].max()),
            }
    return {"single": single, "merged": merged}


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/kl_proxy_analysis_v2", dest="output_dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    BS = args.batch_size

    model = build_model(device)

    torch.manual_seed(42)
    x_1 = torch.randn(BS, ACTION_HORIZON, ACTION_DIM_INTERNAL, dtype=torch.bfloat16, device=device)

    # ── 1. 每个任务加载真实观测，用 base 权重算出 base 的速度轨迹/终点动作 ─────
    logger.info("=== 加载真实观测（moe-demo-clean）并采集 base 的 Euler 轨迹 ===")
    real_obs: dict[str, dict] = {}
    base_vels: dict[str, np.ndarray] = {}
    base_action: dict[str, np.ndarray] = {}
    base_x_traj: dict[str, list] = {}
    base_t_vals: dict[str, list] = {}

    for task in TASK_ORDER:
        real_obs[task] = load_real_obs_by_name(TASK_NAMES[task], BS)
        logger.info(f"  {task}: instr[0]={real_obs[task]['task_descriptions'][0]!r}")

    load_weights(model, MOE_SFT20K)
    model.eval()
    for task in TASK_ORDER:
        pfx, pkv, st = build_prefix_cache(model, real_obs[task], device)
        vels, action, x_traj, t_vals = run_euler_and_collect_velocities(model, pfx, pkv, st, x_1, device)
        base_vels[task] = vels
        base_action[task] = action
        base_x_traj[task] = x_traj
        base_t_vals[task] = t_vals
    logger.info("  base done for all tasks.")

    # ── 2. 5 个 PPO 权重，各自在自己任务的真实观测上算 KL 代理 ────────────────
    logger.info("=== 计算 PPO KL 代理 ===")
    ppo_kl: dict[str, dict] = {}
    for task in TASK_ORDER:
        out = compute_proxies_on_obs(
            model, PPO_WEIGHTS[task], real_obs[task], x_1, device,
            base_vels[task], base_action[task], base_x_traj[task], base_t_vals[task],
        )
        ppo_kl[task] = out
        logger.info(
            f"  {task}: vel_div_self={out['vel_div_self']:.5f}  "
            f"vel_div_same={out['vel_div_same']:.5f}  action_div={out['action_div']:.5f}"
        )

    # ── 3. 10 对融合权重：分别在 A、B 各自真实观测上算 KL 代理 ────────────────
    logger.info("=== 计算 merged KL 代理（分别在 A/B 各自真实观测上）===")
    hs_baseline = load_hiddensim_baseline()

    rows = []
    for ta, tb in PAIRS:
        merged_path = MERGED_WEIGHTS[(ta, tb)]

        out_a = compute_proxies_on_obs(
            model, merged_path, real_obs[ta], x_1, device,
            base_vels[ta], base_action[ta], base_x_traj[ta], base_t_vals[ta],
        )
        out_b = compute_proxies_on_obs(
            model, merged_path, real_obs[tb], x_1, device,
            base_vels[tb], base_action[tb], base_x_traj[tb], base_t_vals[tb],
        )

        kl_A, kl_B = ppo_kl[ta]["vel_div_self"], ppo_kl[tb]["vel_div_self"]
        act_A, act_B = ppo_kl[ta]["action_div"], ppo_kl[tb]["action_div"]

        kl_excess_a = out_a["vel_div_self"] - ALPHA_MERGE * (kl_A + kl_B)
        kl_excess_b = out_b["vel_div_self"] - ALPHA_MERGE * (kl_A + kl_B)
        act_excess_a = out_a["action_div"] - ALPHA_MERGE * (act_A + act_B)
        act_excess_b = out_b["action_div"] - ALPHA_MERGE * (act_A + act_B)

        delta_a = EVAL_RESULTS[(ta, tb)].get(ta, float("nan")) - SINGLE_TASK_BASELINES[ta]
        delta_b = EVAL_RESULTS[(ta, tb)].get(tb, float("nan")) - SINGLE_TASK_BASELINES[tb]
        avg_delta = (delta_a + delta_b) / 2

        hs = hs_baseline["merged"].get((ta, tb), {})
        hs_a_solo = hs_baseline["single"].get(ta, float("nan"))
        hs_b_solo = hs_baseline["single"].get(tb, float("nan"))

        rows.append({
            "pair": f"{ta}+{tb}", "ta": ta, "tb": tb,
            "kl_A": kl_A, "kl_B": kl_B,
            "kl_merged_on_ta": out_a["vel_div_self"], "kl_merged_on_tb": out_b["vel_div_self"],
            "kl_same_merged_on_ta": out_a["vel_div_same"], "kl_same_merged_on_tb": out_b["vel_div_same"],
            "kl_excess_on_ta": kl_excess_a, "kl_excess_on_tb": kl_excess_b,
            "act_A": act_A, "act_B": act_B,
            "act_merged_on_ta": out_a["action_div"], "act_merged_on_tb": out_b["action_div"],
            "act_excess_on_ta": act_excess_a, "act_excess_on_tb": act_excess_b,
            "hs_A_solo": hs_a_solo, "hs_B_solo": hs_b_solo,
            "hs_merged_on_ta": hs.get("on_ta", float("nan")), "hs_merged_on_tb": hs.get("on_tb", float("nan")),
            "delta_a": delta_a, "delta_b": delta_b, "avg_delta": avg_delta,
        })
        logger.info(
            f"  {ta}+{tb}: kl_merged(A-obs/B-obs)={out_a['vel_div_self']:.5f}/{out_b['vel_div_self']:.5f}  "
            f"act_merged(A-obs/B-obs)={out_a['action_div']:.5f}/{out_b['action_div']:.5f}  "
            f"avg_delta={avg_delta:+.2f}pp"
        )

    # ── 4. 写 CSV ─────────────────────────────────────────────────────────────
    with open(out_dir / "kl_proxy_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    logger.info("  saved: kl_proxy_results.csv")

    # ── 5. Pearson r：各预测指标 vs avg_delta（都用 avg(on_ta, on_tb)）────────
    logger.info("\n===== Pearson r vs AvgΔ性能 =====")
    y = np.array([r["avg_delta"] for r in rows])
    valid = ~np.isnan(y)

    def avg_pair(key_a, key_b):
        return np.array([(r[key_a] + r[key_b]) / 2 for r in rows])

    metrics = {
        "kl_A+kl_B (expert vel_div sum)": np.array([r["kl_A"] + r["kl_B"] for r in rows]),
        "kl_merged (self-traj, avg on A/B obs)": avg_pair("kl_merged_on_ta", "kl_merged_on_tb"),
        "kl_same_merged (open-loop, avg on A/B obs)": avg_pair("kl_same_merged_on_ta", "kl_same_merged_on_tb"),
        "kl_excess (avg on A/B obs)": avg_pair("kl_excess_on_ta", "kl_excess_on_tb"),
        "act_A+act_B (expert action_div sum)": np.array([r["act_A"] + r["act_B"] for r in rows]),
        "act_merged (avg on A/B obs)": avg_pair("act_merged_on_ta", "act_merged_on_tb"),
        "act_excess (avg on A/B obs)": avg_pair("act_excess_on_ta", "act_excess_on_tb"),
        "HiddenSim merged (avg on A/B obs, from euler_v2)": avg_pair("hs_merged_on_ta", "hs_merged_on_tb"),
    }

    results_r = []
    for name, x_arr in metrics.items():
        ok = valid & ~np.isnan(x_arr)
        r = float("nan") if ok.sum() < 3 else float(np.corrcoef(x_arr[ok], y[ok])[0, 1])
        results_r.append((name, r))
        logger.info(f"  {name:<48}  r = {r:+.3f}")

    # ── 6. 可视化：散点图 + |r| 对比柱状图 ─────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_metrics = len(results_r)
    ncols = (n_metrics + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))
    axes = axes.flatten()

    for idx, (name, r) in enumerate(results_r):
        ax = axes[idx]
        xm = metrics[name]
        ok = valid & ~np.isnan(xm)
        sc = ax.scatter(xm[ok], y[ok], c=y[ok], cmap="RdYlGn", vmin=-30, vmax=10, s=100, zorder=3)
        plt.colorbar(sc, ax=ax, label="Avg Δ性能 (pp)")
        for ri, rw in enumerate(rows):
            if ok[ri]:
                ax.annotate(rw["pair"], (xm[ri], y[ri]), textcoords="offset points",
                            xytext=(4, 3), fontsize=7)
        if ok.sum() >= 3:
            coeffs = np.polyfit(xm[ok], y[ok], 1)
            xl = np.linspace(xm[ok].min(), xm[ok].max(), 100)
            ax.plot(xl, np.polyval(coeffs, xl), "k--", linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("Avg Δ性能 (pp)", fontsize=8)
        ax.set_title(f"r = {r:+.3f}\n{name}", fontsize=8)
        ax.grid(alpha=0.25)

    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("KL Proxy vs HiddenSim（真实观测修正版）：与融合掉分的 Pearson r 对比", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "kl_vs_hiddensim_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  saved: kl_vs_hiddensim_scatter.png")

    names = [n for n, _ in results_r]
    rs = [r for _, r in results_r]
    colors = ["#1f77b4" if "HiddenSim" in n else "#d62728" for n in names]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(names)), [abs(r) if r == r else 0 for r in rs], color=colors, alpha=0.8)
    for bar, r in zip(bars, rs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"r={r:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|Pearson r| with AvgΔ性能", fontsize=10)
    ax.set_title("预测融合掉分的能力对比：KL proxy（红）vs HiddenSim（蓝）\n|r| 越高 = 预测能力越强",
                 fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "predictor_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  saved: predictor_comparison_bar.png")

    logger.info(f"\n完成。输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
