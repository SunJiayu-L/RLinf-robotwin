#!/usr/bin/env python3
"""Generate bbh/pal/par step50 task-arithmetic grid weights."""

from __future__ import annotations

import logging
import pathlib

import torch

from task_arithmetic_rlinf import (
    PI05_PPO_TRAINABLE_PREFIXES,
    load_pt,
    task_arithmetic,
)

logger = logging.getLogger(__name__)

REPO_PATH = pathlib.Path("/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin")
MERGED_DIR = pathlib.Path("/mnt/petrelfs/muyao/jiayusun/merged_weights")

BASE = (
    "/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/"
    "pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors"
)
PPO = {
    "bbh": (
        REPO_PATH
        / "logs/20260706-23:57:25-moe_sft20k_beat_block_hammer_ppo100"
        / "moe_sft20k_beat_block_hammer_ppo100/checkpoints"
        / "global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "pal": (
        REPO_PATH
        / "logs/20260706-23:28:22-moe_sft20k_place_a2b_left_ppo100"
        / "moe_sft20k_place_a2b_left_ppo100/checkpoints"
        / "global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "par": (
        REPO_PATH
        / "logs/20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100"
        / "moe_sft20k_place_a2b_right_ppo100/checkpoints"
        / "global_step_50/actor/model_state_dict/full_weights.pt"
    ),
}

GRID = [
    ("grid_s50_bbh05_pal05_par05", (0.5, 0.5, 0.5)),
    ("grid_s50_bbh07_pal05_par05", (0.7, 0.5, 0.5)),
    ("grid_s50_bbh05_pal07_par05", (0.5, 0.7, 0.5)),
    ("grid_s50_bbh09_pal05_par05", (0.9, 0.5, 0.5)),
    ("grid_s50_bbh07_pal07_par05", (0.7, 0.7, 0.5)),
    ("grid_s50_bbh05_pal05_par07", (0.5, 0.5, 0.7)),
    ("grid_s50_bbh07_pal05_par07", (0.7, 0.5, 0.7)),
    ("grid_s50_bbh09_pal07_par05", (0.9, 0.7, 0.5)),
    ("grid_s50_bbh05_pal07_par07", (0.5, 0.7, 0.7)),
    ("grid_s50_bbh09_pal05_par07", (0.9, 0.5, 0.7)),
    ("grid_s50_bbh07_pal07_par07", (0.7, 0.7, 0.7)),
    ("grid_s50_bbh09_pal07_par07", (0.9, 0.7, 0.7)),
]

EXISTING_EQUIVALENTS = {
    "grid_s50_bbh07_pal07_par07": MERGED_DIR / "moe_3task_bbh_pal_par_step50_a07/full_weights.pt",
    "grid_s50_bbh09_pal07_par07": MERGED_DIR / "moe_3task_bbh09_pal07_par07_step50/full_weights.pt",
}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_sd = load_pt(BASE)
    ft_sds = [load_pt(PPO[name]) for name in ("bbh", "pal", "par")]

    for label, alphas in GRID:
        existing = EXISTING_EQUIVALENTS.get(label)
        if existing is not None and existing.exists():
            logger.info("skip %s; using existing equivalent %s", label, existing)
            continue

        output = MERGED_DIR / label / "full_weights.pt"
        if output.exists():
            logger.info("skip %s; output already exists: %s", label, output)
            continue

        logger.info("merging %s alphas=%s -> %s", label, alphas, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        merged = task_arithmetic(
            base_sd,
            ft_sds,
            alpha=list(alphas),
            include_prefixes=PI05_PPO_TRAINABLE_PREFIXES,
        )
        torch.save(merged, output)
        del merged


if __name__ == "__main__":
    main()
