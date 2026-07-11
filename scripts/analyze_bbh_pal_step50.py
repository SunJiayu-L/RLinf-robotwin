#!/usr/bin/env python3
"""beat_block_hammer / place_a2b_left 新一轮训练（更大 batch_size）step 50 checkpoint 的 HiddenSim。

复用 scripts/plot_euler_trajectory_heatmaps.py 里已验证正确的方法：
真实观测（moe-demo-clean 录制数据）+ Euler 轨迹，同时报告 self-trajectory（闭环）
和 same-trajectory（开环，固定 base 的 x_t）两种口径。

两个任务各自独立评测（各自的真实观测、各自的 base Euler 轨迹），不做合并分析。

输出：results/bbh_pal_step50/
  <task>_step50_vs_base.png/.npz
"""
from __future__ import annotations

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
    TASK_NAMES,
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

PPO_WEIGHTS = {
    "bbh": (
        f"{REPO_PATH}/logs/20260706-23:57:25-moe_sft20k_beat_block_hammer_ppo100"
        f"/moe_sft20k_beat_block_hammer_ppo100/checkpoints"
        f"/global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "pal": (
        f"{REPO_PATH}/logs/20260706-23:28:22-moe_sft20k_place_a2b_left_ppo100"
        f"/moe_sft20k_place_a2b_left_ppo100/checkpoints"
        f"/global_step_50/actor/model_state_dict/full_weights.pt"
    ),
}
TASKS = ["bbh", "pal"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = pathlib.Path("results/bbh_pal_step50")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    BS = 4

    model = build_model(device)
    torch.manual_seed(42)
    x_1 = torch.randn(BS, ACTION_HORIZON, ACTION_DIM_INTERNAL, dtype=torch.bfloat16, device=device)
    collector = HiddenStateCollector()

    logger.info("=== 加载真实观测 ===")
    real_obs = {}
    for task in TASKS:
        real_obs[task] = load_real_obs_by_name(TASK_NAMES[task], BS)
        logger.info(f"  {task} ({TASK_NAMES[task]}): instr[0]={real_obs[task]['task_descriptions'][0]!r}")

    logger.info("=== 加载 SFT-20k base，逐任务采集 Euler 轨迹 ===")
    load_weights(model, MOE_SFT20K)
    model.eval()
    base_step_hs, base_x_traj = {}, {}
    t_vals = None
    for task in TASKS:
        collector.register(model)
        pfx, pkv, st = build_prefix_cache(model, real_obs[task], device)
        step_hs, t_vals, x_traj = collect_euler_hidden_states(model, pfx, pkv, st, x_1, device, collector)
        collector.clear()
        base_step_hs[task] = step_hs
        base_x_traj[task] = x_traj
        logger.info(f"  base@{task}: done")

    logger.info("=== PPO step50（新一轮，更大 batch_size）===")
    for task in TASKS:
        load_weights(model, PPO_WEIGHTS[task])
        model.eval()

        collector.register(model)
        pfx, pkv, st = build_prefix_cache(model, real_obs[task], device)
        self_hs, _, _ = collect_euler_hidden_states(model, pfx, pkv, st, x_1, device, collector)
        same_hs = probe_fixed_trajectory(model, pfx, pkv, st, base_x_traj[task], t_vals, device, collector)
        collector.clear()

        mat_self = compute_hiddensim_matrix(base_step_hs[task], self_hs)
        mat_same = compute_hiddensim_matrix(base_step_hs[task], same_hs)

        li, si = np.unravel_index(np.argmax(mat_self), mat_self.shape)
        logger.info(
            f"  {task}: self mean={mat_self.mean():.4f} max={mat_self.max():.4f} "
            f"@layer={li},t={t_vals[si]:.1f}  |  "
            f"same mean={mat_same.mean():.4f} max={mat_same.max():.4f}"
        )

        np.savez(out_dir / f"{task}_step50_vs_base.npz",
                 self=mat_self, same=mat_same, t_vals=np.array(t_vals))
        plot_single_task(
            mat_self, mat_same, t_vals, f"{task} ({TASK_NAMES[task]}) step50 [new run]",
            real_obs[task]["task_descriptions"][0],
            out_dir / f"{task}_step50_vs_base.png",
        )

    logger.info(f"\n完成。输出目录：{out_dir.resolve()}")


if __name__ == "__main__":
    main()
