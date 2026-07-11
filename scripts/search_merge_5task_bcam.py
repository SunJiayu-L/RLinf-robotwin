#!/usr/bin/env python3
"""Search and merge 5-task BCAM task-arithmetic candidates.

Tasks:
  bbh: beat_block_hammer step50
  ca:  click_alarmclock step50
  cb:  click_bell step50
  mcp: move_can_pot step100
  mpa: move_playingcard_away step50

The full 0.3..0.9 grid has 7^5 = 16807 candidates.  To avoid writing all
weights to disk, this script first scores the grid from the Gram matrix of
trainable task vectors, then saves only selected candidates.
"""

from __future__ import annotations

import csv
import itertools
import logging
import math
import pathlib
from dataclasses import dataclass

import torch

from task_arithmetic_rlinf import (
    PI05_PPO_TRAINABLE_PREFIXES,
    load_pt,
    task_arithmetic,
)

logger = logging.getLogger(__name__)

REPO_PATH = pathlib.Path("/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin")
MERGED_DIR = pathlib.Path("/mnt/petrelfs/muyao/jiayusun/merged_weights")
OUT_DIR = REPO_PATH / "results/5task_bcam_search"

BASE = (
    "/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/"
    "pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors"
)

TASKS = ("bbh", "ca", "cb", "mcp", "mpa")
TASK_FULL_NAMES = {
    "bbh": "beat_block_hammer",
    "ca": "click_alarmclock",
    "cb": "click_bell",
    "mcp": "move_can_pot",
    "mpa": "move_playingcard_away",
}
PPO = {
    "bbh": (
        REPO_PATH
        / "logs/20260707-22:22:21-moe_sft20k_beat_block_hammer_ppo100"
        / "moe_sft20k_beat_block_hammer_ppo100/checkpoints"
        / "global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "ca": (
        REPO_PATH
        / "logs/20260708-13:50:33-moe_sft20k_click_alarmclock_ppo100"
        / "moe_sft20k_click_alarmclock_ppo100/checkpoints"
        / "global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "cb": (
        REPO_PATH
        / "logs/20260708-13:56:28-moe_sft20k_click_bell_ppo100"
        / "moe_sft20k_click_bell_ppo100/checkpoints"
        / "global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "mcp": (
        REPO_PATH
        / "logs/20260706-23:28:23-moe_sft20k_move_can_pot_ppo100"
        / "moe_sft20k_move_can_pot_ppo100/checkpoints"
        / "global_step_100/actor/model_state_dict/full_weights.pt"
    ),
    "mpa": (
        REPO_PATH
        / "logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100"
        / "moe_sft20k_move_playingcard_away_ppo100/checkpoints"
        / "global_step_50/actor/model_state_dict/full_weights.pt"
    ),
}

ALPHA_VALUES = tuple(round(x * 0.1, 1) for x in range(3, 10))
SAVE_LABELS = {
    "uniform_a03",
    "uniform_a04",
    "uniform_a05",
    "best_mean04",
    "best_mean05",
    "best_mean06",
}


@dataclass(frozen=True)
class CandidateScore:
    label: str
    alphas: tuple[float, ...]
    mean_alpha: float
    l2_norm: float
    norm_per_alpha_sum: float
    rel_to_uniform05: float


def _eligible_keys(base_sd: dict[str, torch.Tensor]) -> list[str]:
    return [
        key
        for key in base_sd
        if any(key.startswith(prefix) for prefix in PI05_PPO_TRAINABLE_PREFIXES)
    ]


def _dot_delta(
    base_sd: dict[str, torch.Tensor],
    left_sd: dict[str, torch.Tensor],
    right_sd: dict[str, torch.Tensor],
    keys: list[str],
) -> float:
    total = torch.tensor(0.0, dtype=torch.float64)
    for key in keys:
        base = base_sd[key].to(torch.float32)
        left = left_sd[key].to(torch.float32) - base
        right = right_sd[key].to(torch.float32) - base
        total += torch.dot(left.flatten().to(torch.float64), right.flatten().to(torch.float64))
    return float(total.item())


def _candidate_label(prefix: str, alphas: tuple[float, ...]) -> str:
    alpha_text = "_".join(
        f"{task}{int(round(alpha * 10)):02d}"
        for task, alpha in zip(TASKS, alphas)
    )
    return f"{prefix}_{alpha_text}"


def _score_grid(gram: torch.Tensor) -> list[CandidateScore]:
    uniform05 = torch.tensor([0.5] * len(TASKS), dtype=torch.float64)
    uniform05_norm = math.sqrt(float(uniform05 @ gram @ uniform05))
    rows: list[CandidateScore] = []
    for alphas in itertools.product(ALPHA_VALUES, repeat=len(TASKS)):
        alpha_t = torch.tensor(alphas, dtype=torch.float64)
        norm_sq = float(alpha_t @ gram @ alpha_t)
        l2_norm = math.sqrt(max(norm_sq, 0.0))
        alpha_sum = sum(alphas)
        rows.append(
            CandidateScore(
                label=_candidate_label("grid", alphas),
                alphas=alphas,
                mean_alpha=alpha_sum / len(alphas),
                l2_norm=l2_norm,
                norm_per_alpha_sum=l2_norm / alpha_sum,
                rel_to_uniform05=l2_norm / uniform05_norm,
            )
        )
    return rows


def _select_candidates(scores: list[CandidateScore]) -> list[CandidateScore]:
    by_alpha = {score.alphas: score for score in scores}
    selected: dict[str, CandidateScore] = {}

    for alpha in (0.3, 0.4, 0.5):
        alphas = tuple([alpha] * len(TASKS))
        score = by_alpha[alphas]
        selected[f"uniform_a{int(alpha * 10):02d}"] = CandidateScore(
            label=f"uniform_a{int(alpha * 10):02d}",
            alphas=score.alphas,
            mean_alpha=score.mean_alpha,
            l2_norm=score.l2_norm,
            norm_per_alpha_sum=score.norm_per_alpha_sum,
            rel_to_uniform05=score.rel_to_uniform05,
        )

    for target_mean in (0.4, 0.5, 0.6):
        candidates = [
            score for score in scores
            if abs(score.mean_alpha - target_mean) < 1e-9
        ]
        best = min(candidates, key=lambda x: (x.norm_per_alpha_sum, x.l2_norm, x.alphas))
        label = f"best_mean{int(target_mean * 10):02d}"
        selected[label] = CandidateScore(
            label=label,
            alphas=best.alphas,
            mean_alpha=best.mean_alpha,
            l2_norm=best.l2_norm,
            norm_per_alpha_sum=best.norm_per_alpha_sum,
            rel_to_uniform05=best.rel_to_uniform05,
        )

    return [selected[label] for label in sorted(SAVE_LABELS)]


def _write_csv(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_search_outputs(
    gram: torch.Tensor,
    scores: list[CandidateScore],
    selected: list[CandidateScore],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gram_rows = []
    for i, task_i in enumerate(TASKS):
        row: dict[str, object] = {"task": task_i}
        for j, task_j in enumerate(TASKS):
            row[task_j] = f"{float(gram[i, j]):.8e}"
        gram_rows.append(row)
    _write_csv(OUT_DIR / "task_vector_gram.csv", gram_rows)

    top_rows = []
    for score in sorted(scores, key=lambda x: (x.norm_per_alpha_sum, x.l2_norm))[:200]:
        top_rows.append(
            {
                "label": score.label,
                **{f"alpha_{task}": alpha for task, alpha in zip(TASKS, score.alphas)},
                "mean_alpha": f"{score.mean_alpha:.4f}",
                "l2_norm": f"{score.l2_norm:.8e}",
                "norm_per_alpha_sum": f"{score.norm_per_alpha_sum:.8e}",
                "rel_to_uniform05": f"{score.rel_to_uniform05:.6f}",
            }
        )
    _write_csv(OUT_DIR / "top200_by_norm_per_alpha.csv", top_rows)

    selected_rows = []
    for score in selected:
        selected_rows.append(
            {
                "label": score.label,
                **{f"alpha_{task}": alpha for task, alpha in zip(TASKS, score.alphas)},
                "mean_alpha": f"{score.mean_alpha:.4f}",
                "l2_norm": f"{score.l2_norm:.8e}",
                "norm_per_alpha_sum": f"{score.norm_per_alpha_sum:.8e}",
                "rel_to_uniform05": f"{score.rel_to_uniform05:.6f}",
                "output": str(MERGED_DIR / f"moe_5task_bcam_{score.label}" / "full_weights.pt"),
            }
        )
    _write_csv(OUT_DIR / "selected_candidates.csv", selected_rows)


def _merge_selected(
    base_sd: dict[str, torch.Tensor],
    ft_sds: list[dict[str, torch.Tensor]],
    selected: list[CandidateScore],
) -> None:
    for score in selected:
        output = MERGED_DIR / f"moe_5task_bcam_{score.label}" / "full_weights.pt"
        if output.exists():
            logger.info("skip existing %s", output)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        logger.info("merge %s alphas=%s -> %s", score.label, score.alphas, output)
        merged = task_arithmetic(
            base_sd,
            ft_sds,
            alpha=list(score.alphas),
            include_prefixes=PI05_PPO_TRAINABLE_PREFIXES,
        )
        torch.save(merged, output)
        del merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for task, path in PPO.items():
        if not path.exists():
            raise FileNotFoundError(f"{task}: {path}")

    logger.info("loading base")
    base_sd = load_pt(BASE)
    keys = _eligible_keys(base_sd)
    logger.info("eligible keys: %d / %d", len(keys), len(base_sd))

    ft_sds = []
    for task in TASKS:
        logger.info("loading %s (%s): %s", task, TASK_FULL_NAMES[task], PPO[task])
        ft_sds.append(load_pt(PPO[task]))

    logger.info("computing task-vector Gram matrix")
    gram = torch.zeros((len(TASKS), len(TASKS)), dtype=torch.float64)
    for i in range(len(TASKS)):
        for j in range(i, len(TASKS)):
            val = _dot_delta(base_sd, ft_sds[i], ft_sds[j], keys)
            gram[i, j] = val
            gram[j, i] = val
            logger.info("  dot(%s,%s)=%.8e", TASKS[i], TASKS[j], val)

    scores = _score_grid(gram)
    selected = _select_candidates(scores)
    _save_search_outputs(gram, scores, selected)

    logger.info("selected candidates:")
    for score in selected:
        logger.info(
            "  %s alphas=%s mean=%.2f rel_uniform05=%.4f",
            score.label,
            score.alphas,
            score.mean_alpha,
            score.rel_to_uniform05,
        )

    _merge_selected(base_sd, ft_sds, selected)
    logger.info("done; outputs in %s", OUT_DIR)


if __name__ == "__main__":
    main()
