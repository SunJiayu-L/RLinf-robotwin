#!/usr/bin/env python3
"""仅计算 par+mcp 这一对的 KL proxy，然后聚合全部 10 对、输出 CSV + 图表。

9/10 对的 KL 数值已从 slurm 日志中提取，硬编码于此脚本。
"""
from __future__ import annotations
import argparse, itertools, logging, pathlib, sys
import numpy as np

logger = logging.getLogger(__name__)

REPO_PATH = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin"
MOE_SFT20K = (
    "/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/"
    "pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors"
)
MOE_MODEL_PATH = (
    "/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/"
    "pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000"
)
CONFIG_NAME = "pi05_aloha_moe_demo_clean"
MERGED_DIR = "/mnt/petrelfs/muyao/jiayusun/merged_weights"
PREV_RESULT_DIR = f"{REPO_PATH}/results/pairwise_merge_analysis"

ACTION_DIM_INTERNAL = 32
ACTION_HORIZON = 50
T_STEPS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA_MERGE = 0.7
EULER_STEPS = 5

# ── 已知数值（从 slurm 日志 10204323 提取）──────────────────────────────────────
# PPO KL vs SFT-20k base
KNOWN_PPO_KL = {
    "bbh": {"vel_div": 0.02474, "action_div": 0.06735},
    "pps": {"vel_div": 0.01323, "action_div": 0.13158},
    "pal": {"vel_div": 0.01851, "action_div": 0.32491},
    "par": {"vel_div": 0.00660, "action_div": 0.09663},
    "mcp": {"vel_div": 0.02209, "action_div": 0.06798},
}

# 已知 9/10 merged pair KL 数值（从日志提取）
# key: (ta, tb) → {kl_merged, act_merged}
KNOWN_MERGED_KL: dict[tuple[str, str], dict] = {
    ("bbh", "pps"): {"kl_merged": 0.02015, "act_merged": 0.10354},
    ("bbh", "pal"): {"kl_merged": 0.02406, "act_merged": 0.25115},
    ("bbh", "par"): {"kl_merged": 0.01912, "act_merged": 0.05706},
    ("bbh", "mcp"): {"kl_merged": 0.02791, "act_merged": 0.07368},
    ("pps", "pal"): {"kl_merged": 0.01807, "act_merged": 0.21570},
    ("pps", "par"): {"kl_merged": 0.00942, "act_merged": 0.05466},
    ("pps", "mcp"): {"kl_merged": 0.01781, "act_merged": 0.11679},
    ("pal", "par"): {"kl_merged": 0.01324, "act_merged": 0.28041},
    ("pal", "mcp"): {"kl_merged": 0.02048, "act_merged": 0.28320},
    # ("par", "mcp") → to be computed below
}

EVAL_RESULTS = {
    ("bbh", "pps"): {"bbh": 63.4, "pps": 76.9},
    ("bbh", "pal"): {"bbh": 55.7, "pal": 48.2},
    ("bbh", "par"): {"bbh": 76.5, "par": 59.2},
    ("bbh", "mcp"): {"bbh": 39.9, "mcp": 67.7},
    ("pps", "pal"): {"pps": 54.1, "pal": 66.6},
    ("pps", "par"): {"pps": 70.7, "par": 70.5},
    ("pps", "mcp"): {"pps": 72.0, "mcp": 46.4},
    ("pal", "par"): {"pal": 64.5, "par": 56.0},
    ("pal", "mcp"): {"pal": 62.4, "mcp": 40.8},
    ("par", "mcp"): {"par": 58.1, "mcp": 62.6},
}
SINGLE_TASK_BASELINES = {
    "bbh": 86.1, "pps": 70.7, "pal": 51.9, "par": 57.1, "mcp": 63.6,
}
TASK_ORDER = ["bbh", "pps", "pal", "par", "mcp"]
PAIRS = list(itertools.combinations(TASK_ORDER, 2))


def build_model(device: str):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "model_path": MOE_MODEL_PATH, "model_type": "openpi",
        "precision": "bfloat16", "num_action_chunks": ACTION_HORIZON,
        "action_dim": 14, "num_steps": EULER_STEPS,
        "is_lora": False, "load_to_device": False,
        "openpi": {
            "config_name": CONFIG_NAME, "num_images_in_input": 3,
            "noise_level": 0.3, "detach_critic_input": True,
            "add_value_head": True, "train_expert_only": True,
        },
    })
    from rlinf.scheduler.worker.worker import Worker as _W
    from rlinf.scheduler.hardware import AcceleratorUtil
    _W.torch_platform = AcceleratorUtil.get_torch_platform(AcceleratorUtil.get_accelerator_type())
    _W.torch_device_type = AcceleratorUtil.get_device_type(AcceleratorUtil.get_accelerator_type())
    from rlinf.models.embodiment.openpi import get_model
    model = get_model(cfg, torch_dtype=torch.bfloat16)
    model.eval(); model.to(device)
    return model


def load_weights(model, path: str) -> bool:
    import torch, pathlib
    from safetensors.torch import load_file
    p = pathlib.Path(path)
    if not p.exists():
        logger.warning(f"NOT FOUND: {path}"); return False
    sd = load_file(path) if p.suffix == ".safetensors" else torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(torch.bfloat16)
    return True


def make_obs(batch_size: int, seed: int = 42):
    import numpy as np, torch
    rng = np.random.default_rng(seed)
    H, W = 224, 224
    return {
        "main_images": torch.from_numpy(rng.integers(0,256,(H,W,3),dtype=np.uint8)[None].repeat(batch_size,0)),
        "wrist_images": torch.from_numpy(
            np.stack([np.stack([rng.integers(0,256,(H,W,3),dtype=np.uint8)]*2)]*batch_size)),
        "states": torch.from_numpy(rng.uniform(-1,1,(14,)).astype(np.float32)[None].repeat(batch_size,0)),
        "task_descriptions": ["beat block hammer"]*batch_size,
        "extra_view_images": None,
    }


def build_prefix_cache(model, obs, device: str):
    import openpi.models.model as _m
    import torch
    to_proc = model.obs_processor(obs)
    processed = model.input_transform(to_proc, transpose=False)
    processed = model.precision_processor(processed)
    observation = _m.Observation.from_dict(processed)
    with torch.no_grad():
        imgs, img_masks, lang_tok, lang_masks, state = model._preprocess_observation(observation, train=False)
        _, pfx_pads, pkv = model._build_prefix_cache(imgs, img_masks, lang_tok, lang_masks)
    return pfx_pads, pkv, state


def _get_velocity(model, state, x_t, t_tensor, pfx_pads, pkv):
    import torch
    suffix_out = model.get_suffix_out(state, pfx_pads, pkv, x_t, t_tensor)
    proj_dtype = model.action_out_proj.weight.dtype
    return model.action_out_proj(suffix_out.to(proj_dtype)).float()


def compute_vels(model, pfx_pads, pkv, state, x_t_fixed, device):
    import torch, numpy as np
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    vels = []
    for t_val in T_STEPS:
        t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
        v = _get_velocity(model, state, x_t_fixed, t_tensor, pfx_pads, pkv)
        vels.append(v.cpu().numpy())
    return np.stack(vels, axis=0)


def euler_denoise(model, pfx_pads, pkv, state, x_1, device, num_steps=EULER_STEPS):
    import torch
    B = state.shape[0]
    dtype = model.action_in_proj.weight.dtype
    x = x_1.clone().to(dtype).to(device)
    dt = -1.0 / num_steps
    for i in range(num_steps):
        t_val = 1.0 + i * dt
        t_tensor = torch.full((B,), t_val, dtype=dtype, device=device)
        v = _get_velocity(model, state, x, t_tensor, pfx_pads, pkv)
        x = x + dt * v.to(dtype)
    return x.float().cpu().numpy()


def norm_l2_div(a, b):
    diff = np.linalg.norm((a - b).reshape(-1))
    return float(diff / (np.linalg.norm(a.reshape(-1)) + np.linalg.norm(b.reshape(-1)) + 1e-8))


def main():
    import torch
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/kl_proxy_analysis")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    sys.path.insert(0, REPO_PATH)
    model = build_model(device)

    torch.manual_seed(42)
    BS = args.batch_size
    x_t_fixed = torch.randn(BS, ACTION_HORIZON, ACTION_DIM_INTERNAL, dtype=torch.bfloat16, device=device)
    x_1_fixed = torch.randn(BS, ACTION_HORIZON, ACTION_DIM_INTERNAL, dtype=torch.bfloat16, device=device)
    obs = make_obs(BS)

    # Load base and compute base vels/action
    logger.info("=== Loading SFT-20k base ===")
    load_weights(model, MOE_SFT20K)
    model.eval()
    pfx_b, pkv_b, st_b = build_prefix_cache(model, obs, device)
    base_vels   = compute_vels(model, pfx_b, pkv_b, st_b, x_t_fixed, device)
    base_action = euler_denoise(model, pfx_b, pkv_b, st_b, x_1_fixed, device)
    logger.info(f"  base_vels shape={base_vels.shape}")

    # Compute par+mcp merged KL
    logger.info("=== Computing par+mcp merged KL ===")
    merged_path = f"{MERGED_DIR}/pair_place_a2b_right__move_can_pot_a07/full_weights.pt"
    load_weights(model, merged_path)
    model.eval()
    pfx_m, pkv_m, st_m = build_prefix_cache(model, obs, device)
    with torch.no_grad():
        m_vels   = compute_vels(model, pfx_m, pkv_m, st_m, x_t_fixed, device)
        m_action = euler_denoise(model, pfx_m, pkv_m, st_m, x_1_fixed, device)
    kl_m  = norm_l2_div(base_vels, m_vels)
    act_m = norm_l2_div(base_action, m_action)
    logger.info(f"  par+mcp: kl_merged={kl_m:.5f}, act_merged={act_m:.5f}")
    KNOWN_MERGED_KL[("par", "mcp")] = {"kl_merged": kl_m, "act_merged": act_m}

    # Assemble full results
    rows = []
    for ta, tb in PAIRS:
        kl_A  = KNOWN_PPO_KL[ta]["vel_div"]
        kl_B  = KNOWN_PPO_KL[tb]["vel_div"]
        act_A = KNOWN_PPO_KL[ta]["action_div"]
        act_B = KNOWN_PPO_KL[tb]["action_div"]
        km    = KNOWN_MERGED_KL[(ta, tb)]
        kl_merged  = km["kl_merged"]
        act_merged = km["act_merged"]
        kl_excess  = kl_merged  - ALPHA_MERGE * (kl_A  + kl_B)
        act_excess = act_merged - ALPHA_MERGE * (act_A + act_B)
        delta_a   = EVAL_RESULTS[(ta,tb)].get(ta, float("nan")) - SINGLE_TASK_BASELINES[ta]
        delta_b   = EVAL_RESULTS[(ta,tb)].get(tb, float("nan")) - SINGLE_TASK_BASELINES[tb]
        avg_delta = (delta_a + delta_b) / 2
        rows.append({
            "pair": f"{ta}+{tb}", "ta": ta, "tb": tb,
            "kl_A": kl_A, "kl_B": kl_B, "kl_merged": kl_merged, "kl_excess": kl_excess,
            "kl_sum_ab": kl_A+kl_B,
            "act_A": act_A, "act_B": act_B, "act_merged": act_merged, "act_excess": act_excess,
            "act_sum_ab": act_A+act_B,
            "delta_a": delta_a, "delta_b": delta_b, "avg_delta": avg_delta,
        })

    # Save CSV
    import csv
    with open(out_dir / "kl_proxy_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    logger.info("  saved: kl_proxy_results.csv")

    # Pearson r analysis
    logger.info("\n===== Pearson r vs AvgΔ =====")
    y = np.array([r["avg_delta"] for r in rows])
    metrics = {
        "kl_A+kl_B (expert KL sum)":        np.array([r["kl_A"]+r["kl_B"]   for r in rows]),
        "kl_merged (merged vel div)":        np.array([r["kl_merged"]         for r in rows]),
        "kl_excess (conflict KL)":           np.array([r["kl_excess"]         for r in rows]),
        "act_sum_ab (expert action KL sum)": np.array([r["act_sum_ab"]        for r in rows]),
        "act_merged (merged ODE div)":       np.array([r["act_merged"]        for r in rows]),
        "act_excess (conflict action KL)":   np.array([r["act_excess"]        for r in rows]),
    }
    # Load prior HiddenSim
    hs_ppo_path = pathlib.Path(PREV_RESULT_DIR) / "hiddensim_ppo_pairs.npy"
    if hs_ppo_path.exists():
        hs_ppo = np.load(hs_ppo_path).mean(axis=(1,2,3,4))
        hs_va  = np.load(pathlib.Path(PREV_RESULT_DIR)/"hiddensim_merged_vs_a.npy").mean(axis=(1,2,3,4))
        hs_vb  = np.load(pathlib.Path(PREV_RESULT_DIR)/"hiddensim_merged_vs_b.npy").mean(axis=(1,2,3,4))
        metrics["HS(PPO_A,PPO_B) [HiddenSim]"]          = hs_ppo
        metrics["HS(merged,A)+HS(merged,B) [HiddenSim]"] = hs_va + hs_vb

    results_r = []
    for name, x_arr in metrics.items():
        ok = ~np.isnan(x_arr) & ~np.isnan(y)
        r = float(np.corrcoef(x_arr[ok], y[ok])[0,1]) if ok.sum() >= 3 else float("nan")
        results_r.append((name, r))
        logger.info(f"  {name:<48}  r = {r:+.3f}")

    # Plots
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # scatter plots
    n_m = len(results_r)
    ncols = (n_m + 1) // 2
    fig, axes = plt.subplots(2, ncols, figsize=(6*ncols, 10))
    axes = axes.flatten()
    for idx, (name, r) in enumerate(results_r):
        ax = axes[idx]
        xm = metrics[name]
        ok = ~np.isnan(xm) & ~np.isnan(y)
        sc = ax.scatter(xm[ok], y[ok], c=y[ok], cmap="RdYlGn", vmin=-50, vmax=20, s=100, zorder=3)
        plt.colorbar(sc, ax=ax, label="Avg Δ (pp)")
        for ri, rw in enumerate(rows):
            if ok[ri]:
                ax.annotate(rw["pair"], (xm[ri], y[ri]),
                            textcoords="offset points", xytext=(4,3), fontsize=7)
        if ok.sum() >= 3:
            coeffs = np.polyfit(xm[ok], y[ok], 1)
            xl = np.linspace(xm[ok].min(), xm[ok].max(), 100)
            ax.plot(xl, np.polyval(coeffs, xl), "k--", linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(name.split("(")[0].strip(), fontsize=8)
        ax.set_ylabel("Avg Δ (pp)", fontsize=8)
        ax.set_title(f"r = {r:+.3f}\n{name}", fontsize=8)
        ax.grid(alpha=0.25)
    for idx in range(n_m, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle("KL Proxy vs HiddenSim：与融合掉分的 Pearson r 对比", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "kl_vs_hiddensim_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  saved: kl_vs_hiddensim_scatter.png")

    # bar chart
    names = [n for n, _ in results_r]
    rs    = [r for _, r in results_r]
    colors = ["#d62728" if any(k in n.lower() for k in ["kl","act_"]) else "#1f77b4"
              for n in names]
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(range(len(names)), [abs(r) if not np.isnan(r) else 0 for r in rs],
                  color=colors, alpha=0.8)
    for i, (bar, r) in enumerate(zip(bars, rs)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"r={r:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split("[")[0].strip() for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("|Pearson r| with AvgΔ", fontsize=10)
    ax.set_title("预测融合掉分的能力对比：KL proxy（红）vs HiddenSim（蓝）\n|r| 越高 = 预测能力越强", fontsize=11)
    ax.set_ylim(0, 1); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "predictor_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  saved: predictor_comparison_bar.png")

    # Summary table
    logger.info("\n===== 汇总表（按 avg_delta 排序）=====")
    logger.info(f"{'Pair':<12} {'kl_A+kl_B':>10} {'kl_merged':>10} {'kl_excess':>10} "
                f"{'act_sum':>10} {'act_merged':>11} {'AvgΔ':>7}")
    logger.info("-" * 75)
    for r in sorted(rows, key=lambda x: x["avg_delta"]):
        logger.info(f"{r['pair']:<12} {r['kl_A']+r['kl_B']:>10.5f} {r['kl_merged']:>10.5f} "
                    f"{r['kl_excess']:>10.5f} {r['act_A']+r['act_B']:>10.5f} "
                    f"{r['act_merged']:>11.5f} {r['avg_delta']:>+7.1f}")
    logger.info(f"\nDone. Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
