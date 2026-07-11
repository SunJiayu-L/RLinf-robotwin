#!/usr/bin/env python3
"""move_playingcard_away 多个 PPO checkpoint（50/100/150 步）vs SFT-20k base 的 HiddenSim。

复用 scripts/plot_euler_trajectory_heatmaps.py 里已验证正确的方法：
真实观测（moe-demo-clean 录制数据）+ Euler 轨迹，同时报告 self-trajectory（闭环）
和 same-trajectory（开环，固定 base 的 x_t）两种口径。

输出：results/move_playingcard_away_steps/
  mpa_step<N>_vs_base.png/.npz   每个 PPO step 的单任务热力图（self | same 两个面板）
  mpa_summary.json               三个 step 的 mean/max 汇总，供跨 step 对比
"""
from __future__ import annotations

import argparse
import json
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
    MOE_SFT20K,
    HiddenStateCollector,
    build_model,
    build_prefix_cache,
    collect_euler_hidden_states,
    compute_hiddensim_matrix,
    load_real_obs_by_name,
    load_weights,
    plot_single_task,
    probe_fixed_trajectory,
)

TASK_NAME = "move_playingcard_away"
CKPT_DIR = (
    f"{REPO_PATH}/logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100"
    f"/moe_sft20k_move_playingcard_away_ppo100/checkpoints"
)
PPO_STEPS = [50, 100, 150]
PPO_WEIGHTS = {
    s: f"{CKPT_DIR}/global_step_{s}/actor/model_state_dict/full_weights.pt" for s in PPO_STEPS
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/move_playingcard_away_steps")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    BS = args.batch_size

    model = build_model(device)
    torch.manual_seed(42)
    x_1 = torch.randn(BS, ACTION_HORIZON, ACTION_DIM_INTERNAL, dtype=torch.bfloat16, device=device)
    collector = HiddenStateCollector()

    logger.info(f"=== 加载真实观测：{TASK_NAME} ===")
    obs = load_real_obs_by_name(TASK_NAME, BS)
    logger.info(f"  instr[0]={obs['task_descriptions'][0]!r}")

    logger.info("=== 加载 SFT-20k base，采集 Euler 轨迹（含 x_t 序列）===")
    load_weights(model, MOE_SFT20K)
    model.eval()
    collector.register(model)
    pfx_b, pkv_b, st_b = build_prefix_cache(model, obs, device)
    base_step_hs, t_vals, base_x_traj = collect_euler_hidden_states(
        model, pfx_b, pkv_b, st_b, x_1, device, collector)
    collector.clear()
    logger.info(f"  base done. t_vals={t_vals}")

    summary = {}
    for step in PPO_STEPS:
        logger.info(f"=== PPO step {step} ===")
        load_weights(model, PPO_WEIGHTS[step])
        model.eval()

        collector.register(model)
        pfx_p, pkv_p, st_p = build_prefix_cache(model, obs, device)
        self_hs, _, _ = collect_euler_hidden_states(model, pfx_p, pkv_p, st_p, x_1, device, collector)
        same_hs = probe_fixed_trajectory(
            model, pfx_p, pkv_p, st_p, base_x_traj, t_vals, device, collector)
        collector.clear()

        mat_self = compute_hiddensim_matrix(base_step_hs, self_hs)
        mat_same = compute_hiddensim_matrix(base_step_hs, same_hs)

        li_s, si_s = np.unravel_index(np.argmax(mat_self), mat_self.shape)
        li_o, si_o = np.unravel_index(np.argmax(mat_same), mat_same.shape)
        logger.info(
            f"  step {step}: self mean={mat_self.mean():.4f} max={mat_self.max():.4f} "
            f"@layer={li_s},t={t_vals[si_s]:.1f}  |  "
            f"same mean={mat_same.mean():.4f} max={mat_same.max():.4f} @layer={li_o},t={t_vals[si_o]:.1f}"
        )

        np.savez(out_dir / f"mpa_step{step}_vs_base.npz",
                 self=mat_self, same=mat_same, t_vals=np.array(t_vals))
        plot_single_task(
            mat_self, mat_same, t_vals, f"move_playingcard_away step {step}",
            obs["task_descriptions"][0],
            out_dir / f"mpa_step{step}_vs_base.png",
        )

        summary[str(step)] = {
            "self_mean": float(mat_self.mean()), "self_max": float(mat_self.max()),
            "self_max_layer": int(li_s), "self_max_t": float(t_vals[si_s]),
            "same_mean": float(mat_same.mean()), "same_max": float(mat_same.max()),
            "same_max_layer": int(li_o), "same_max_t": float(t_vals[si_o]),
        }

    with open(out_dir / "mpa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== 汇总（跨 PPO step）===")
    for step in PPO_STEPS:
        s = summary[str(step)]
        logger.info(f"  step {step}: self_max={s['self_max']:.4f}  same_max={s['same_max']:.4f}")

    logger.info(f"\n完成。输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
