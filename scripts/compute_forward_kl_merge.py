#!/usr/bin/env python3
"""Forward-KL diagnostics for pi0.5 Flow-SDE weight merging.

The metric follows the useful direction from RL's Razor:

    E_obs KL(pi_base(. | obs) || pi_candidate(. | obs))

For pi-RL Flow-SDE, each denoising transition is Gaussian.  This script
therefore computes the analytic KL between the SFT/base transition and a
candidate transition on the same task observations and base Flow-SDE
trajectories.  It is intended as an offline filter for weight-fusion
candidates, not as a training loss.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

REPO_PATH = "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin"
sys.path.insert(0, REPO_PATH)
sys.path.insert(0, str(pathlib.Path(REPO_PATH) / "scripts"))

from plot_euler_trajectory_heatmaps import (  # noqa: E402
    ACTION_DIM_INTERNAL,
    ACTION_HORIZON,
    EULER_STEPS,
    MERGED_DIR,
    MOE_SFT20K,
    TASK_NAMES,
    TASK_ORDER,
    build_model,
    build_prefix_cache,
    load_real_obs_by_name,
    load_weights,
)
from task_arithmetic_rlinf import (  # noqa: E402
    PI05_PPO_TRAINABLE_PREFIXES,
    load_pt,
    task_arithmetic,
)

TASK_NAMES_EXT = {
    **TASK_NAMES,
    "mpa": "move_playingcard_away",
}

OLD_BEST_PPO_WEIGHTS = {
    "bbh": (
        f"{REPO_PATH}/logs/20260625-11:37:02-moe_sft20k_beat_block_hammer_ppo100"
        "/moe_sft20k_beat_block_hammer_ppo100/checkpoints"
        "/global_step_300/actor/model_state_dict/full_weights.pt"
    ),
    "pps": (
        f"{REPO_PATH}/logs/20260703-21:07:14-moe_sft20k_place_phone_stand_ppo100"
        "/moe_sft20k_place_phone_stand_ppo100/checkpoints"
        "/global_step_300/actor/model_state_dict/full_weights.pt"
    ),
    "pal": (
        f"{REPO_PATH}/logs/20260703-21:07:14-moe_sft20k_place_a2b_left_ppo100"
        "/moe_sft20k_place_a2b_left_ppo100/checkpoints"
        "/global_step_300/actor/model_state_dict/full_weights.pt"
    ),
    "par": (
        f"{REPO_PATH}/logs/20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100"
        "/moe_sft20k_place_a2b_right_ppo100/checkpoints"
        "/global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "mcp": (
        f"{REPO_PATH}/logs/20260703-00:38:41-moe_sft20k_move_can_pot_ppo100"
        "/moe_sft20k_move_can_pot_ppo100/checkpoints"
        "/global_step_400/actor/model_state_dict/full_weights.pt"
    ),
}

STEP_PPO_WEIGHTS = {
    "bbh50": (
        f"{REPO_PATH}/logs/20260706-23:57:25-moe_sft20k_beat_block_hammer_ppo100"
        "/moe_sft20k_beat_block_hammer_ppo100/checkpoints"
        "/global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "pal50": (
        f"{REPO_PATH}/logs/20260706-23:28:22-moe_sft20k_place_a2b_left_ppo100"
        "/moe_sft20k_place_a2b_left_ppo100/checkpoints"
        "/global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "pal100": (
        f"{REPO_PATH}/logs/20260706-23:28:22-moe_sft20k_place_a2b_left_ppo100"
        "/moe_sft20k_place_a2b_left_ppo100/checkpoints"
        "/global_step_100/actor/model_state_dict/full_weights.pt"
    ),
    "par50": OLD_BEST_PPO_WEIGHTS["par"],
    "par200": (
        f"{REPO_PATH}/logs/20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100"
        "/moe_sft20k_place_a2b_right_ppo100/checkpoints"
        "/global_step_200/actor/model_state_dict/full_weights.pt"
    ),
    "mcp50": (
        f"{REPO_PATH}/logs/20260706-23:28:23-moe_sft20k_move_can_pot_ppo100"
        "/moe_sft20k_move_can_pot_ppo100/checkpoints"
        "/global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "mcp100": (
        f"{REPO_PATH}/logs/20260706-23:28:23-moe_sft20k_move_can_pot_ppo100"
        "/moe_sft20k_move_can_pot_ppo100/checkpoints"
        "/global_step_100/actor/model_state_dict/full_weights.pt"
    ),
    "mpa50": (
        f"{REPO_PATH}/logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100"
        "/moe_sft20k_move_playingcard_away_ppo100/checkpoints"
        "/global_step_50/actor/model_state_dict/full_weights.pt"
    ),
    "mpa100": (
        f"{REPO_PATH}/logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100"
        "/moe_sft20k_move_playingcard_away_ppo100/checkpoints"
        "/global_step_100/actor/model_state_dict/full_weights.pt"
    ),
    "mpa150": (
        f"{REPO_PATH}/logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100"
        "/moe_sft20k_move_playingcard_away_ppo100/checkpoints"
        "/global_step_150/actor/model_state_dict/full_weights.pt"
    ),
}

TASK_TO_STEP50_KEY = {
    "bbh": "bbh50",
    "pal": "pal50",
    "par": "par50",
    "mcp": "mcp50",
    "mpa": "mpa50",
}


@dataclass(frozen=True)
class Candidate:
    label: str
    tasks: tuple[str, ...]
    source: str
    path: str | None = None
    ft_keys: tuple[str, ...] = ()
    alphas: tuple[float, ...] = ()
    solo_keys: dict[str, str] | None = None


def _existing_candidates() -> list[Candidate]:
    pairs = {
        "pair_bbh_pps_a07": (("bbh", "pps"), f"{MERGED_DIR}/pair_beat_block_hammer__place_phone_stand_a07/full_weights.pt"),
        "pair_bbh_pal_a07": (("bbh", "pal"), f"{MERGED_DIR}/pair_beat_block_hammer__place_a2b_left_a07/full_weights.pt"),
        "pair_bbh_par_a07": (("bbh", "par"), f"{MERGED_DIR}/pair_beat_block_hammer__place_a2b_right_a07/full_weights.pt"),
        "pair_bbh_mcp_a07": (("bbh", "mcp"), f"{MERGED_DIR}/pair_beat_block_hammer__move_can_pot_a07/full_weights.pt"),
        "pair_pps_pal_a07": (("pps", "pal"), f"{MERGED_DIR}/pair_place_phone_stand__place_a2b_left_a07/full_weights.pt"),
        "pair_pps_par_a07": (("pps", "par"), f"{MERGED_DIR}/pair_place_phone_stand__place_a2b_right_a07/full_weights.pt"),
        "pair_pps_mcp_a07": (("pps", "mcp"), f"{MERGED_DIR}/pair_place_phone_stand__move_can_pot_a07/full_weights.pt"),
        "pair_pal_par_a07": (("pal", "par"), f"{MERGED_DIR}/pair_place_a2b_left__place_a2b_right_a07/full_weights.pt"),
        "pair_pal_mcp_a07": (("pal", "mcp"), f"{MERGED_DIR}/pair_place_a2b_left__move_can_pot_a07/full_weights.pt"),
        "pair_par_mcp_a07": (("par", "mcp"), f"{MERGED_DIR}/pair_place_a2b_right__move_can_pot_a07/full_weights.pt"),
    }
    out = [
        Candidate(label, tasks, "existing_pair_old_best", path, solo_keys={t: f"old_{t}" for t in tasks})
        for label, (tasks, path) in pairs.items()
    ]
    out.extend(
        [
            Candidate(
                "old_5task_a07",
                ("bbh", "pps", "pal", "par", "mcp"),
                "existing_old_best",
                f"{MERGED_DIR}/moe_sft20k_ppo5task_a07/full_weights.pt",
                solo_keys={t: f"old_{t}" for t in ("bbh", "pps", "pal", "par", "mcp")},
            ),
            Candidate(
                "old_5task_chain_bbh_a07",
                ("bbh", "pps", "pal", "par", "mcp"),
                "existing_old_best",
                f"{MERGED_DIR}/moe_sft20k_ppo5task_chain_bbh_a07/full_weights.pt",
                solo_keys={t: f"old_{t}" for t in ("bbh", "pps", "pal", "par", "mcp")},
            ),
            Candidate(
                "s50_bbh07_pal07_par07",
                ("bbh", "pal", "par"),
                "existing_step50",
                f"{MERGED_DIR}/moe_3task_bbh_pal_par_step50_a07/full_weights.pt",
                solo_keys={"bbh": "bbh50", "pal": "pal50", "par": "par50"},
            ),
            Candidate(
                "s50_bbh09_pal07_par07",
                ("bbh", "pal", "par"),
                "existing_step50",
                f"{MERGED_DIR}/moe_3task_bbh09_pal07_par07_step50/full_weights.pt",
                solo_keys={"bbh": "bbh50", "pal": "pal50", "par": "par50"},
            ),
            Candidate(
                "s50_pal07_par07",
                ("pal", "par"),
                "existing_step50",
                f"{MERGED_DIR}/moe_2task_pal07_par07_step50/full_weights.pt",
                solo_keys={"pal": "pal50", "par": "par50"},
            ),
            Candidate(
                "s50_pal07_par07_mpa07",
                ("pal", "par", "mpa"),
                "existing_step50",
                f"{MERGED_DIR}/moe_3task_pal07_par07_mpa07_step50/full_weights.pt",
                solo_keys={"pal": "pal50", "par": "par50", "mpa": "mpa50"},
            ),
            Candidate(
                "s50_pal08_par08_mpa07",
                ("pal", "par", "mpa"),
                "existing_step50",
                f"{MERGED_DIR}/moe_3task_pal08_par08_mpa07_step50/full_weights.pt",
                solo_keys={"pal": "pal50", "par": "par50", "mpa": "mpa50"},
            ),
            Candidate(
                "s50_mpa_pal_par_mcp_a07",
                ("mpa", "pal", "par", "mcp"),
                "existing_step_mix",
                f"{MERGED_DIR}/moe_4task_mpa_pal_par_mcp_s50s50s50s100_a07/full_weights.pt",
                solo_keys={"mpa": "mpa50", "pal": "pal50", "par": "par50", "mcp": "mcp100"},
            ),
        ]
    )
    return [c for c in out if c.path is not None and pathlib.Path(c.path).exists()]


def _grid_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for bbh, pal, par in itertools.product((0.5, 0.7, 0.9), (0.5, 0.7), (0.5, 0.7)):
        label = f"grid_s50_bbh{bbh:.1f}_pal{pal:.1f}_par{par:.1f}".replace(".", "")
        out.append(
            Candidate(
                label=label,
                tasks=("bbh", "pal", "par"),
                source="grid_step50_in_memory",
                ft_keys=("bbh50", "pal50", "par50"),
                alphas=(bbh, pal, par),
                solo_keys={"bbh": "bbh50", "pal": "pal50", "par": "par50"},
            )
        )
    return out


def _validate_model_load(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], label: str) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if "value_head" not in k and "embed_tokens" not in k]
    if missing:
        raise RuntimeError(f"missing keys loading {label}: {missing[:10]}")
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected keys loading {label}: {result.unexpected_keys[:10]}")
    model.to(torch.bfloat16)


def _flow_sde_step_params(model, pfx_pads, pkv, state, x_t, idx: int, device: str):
    dtype = model.action_in_proj.weight.dtype
    timesteps = torch.linspace(1, 1 / EULER_STEPS, EULER_STEPS, device=device, dtype=dtype)
    timesteps = torch.cat([timesteps, torch.zeros(1, device=device, dtype=dtype)])
    noise_level = model._get_noise_level(device=device, dtype=dtype, sample_method="flow_sde")
    denom_timesteps = torch.where(timesteps == 1, timesteps[1], timesteps)
    sigma_ratio = timesteps / (1 - denom_timesteps)
    sigmas = noise_level * torch.sqrt(sigma_ratio)[:-1]

    bsz = state.shape[0]
    x = x_t.to(dtype).to(device)
    t_input_scalar = timesteps[idx]
    next_t_scalar = timesteps[idx + 1]
    delta_scalar = t_input_scalar - next_t_scalar
    t_input = torch.full((bsz,), float(t_input_scalar.item()), dtype=dtype, device=device)
    suffix_out = model.get_suffix_out(state, pfx_pads, pkv, x, t_input)
    proj_dtype = model.action_out_proj.weight.dtype
    v_t = model.action_out_proj(suffix_out.to(proj_dtype))

    t_expanded = t_input_scalar.view(1, 1, 1).expand_as(x)
    next_t = next_t_scalar.view(1, 1, 1).expand_as(x)
    delta = delta_scalar.view(1, 1, 1).expand_as(x)
    sigma_i = sigmas[idx].view(1, 1, 1).expand_as(x)

    v_t = v_t.to(dtype)
    x0_pred = x - v_t * t_expanded
    x1_pred = x + v_t * (1 - t_expanded)
    x0_weight = torch.ones_like(t_expanded) - next_t
    x1_weight = next_t - sigma_i**2 * delta / (2 * t_expanded)
    std = torch.sqrt(delta) * sigma_i
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    return mean.float().detach(), std.float().detach()


def _sample_base_sde_trajectory(
    model,
    pfx_pads,
    pkv,
    state,
    seed: int,
    batch_size: int,
    device: str,
):
    dtype = model.action_in_proj.weight.dtype
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(
        batch_size,
        ACTION_HORIZON,
        ACTION_DIM_INTERNAL,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    timesteps = torch.linspace(1, 1 / EULER_STEPS, EULER_STEPS, device=device, dtype=dtype)
    x_traj: list[torch.Tensor] = []
    params = []
    t_vals: list[float] = []

    with torch.no_grad():
        for idx in range(EULER_STEPS):
            x_traj.append(x.detach().clone())
            t_vals.append(round(float(timesteps[idx].item()), 4))
            mean, std = _flow_sde_step_params(model, pfx_pads, pkv, state, x, idx, device)
            params.append((mean, std))
            noise = torch.randn(
                x.shape,
                generator=generator,
                dtype=torch.float32,
                device=device,
            )
            x = (mean + noise * std).to(dtype)
    return x_traj, t_vals, params


def _flow_sde_params(model, pfx_pads, pkv, state, x_traj, device: str):
    params = []
    with torch.no_grad():
        for idx, x_t in enumerate(x_traj):
            params.append(_flow_sde_step_params(model, pfx_pads, pkv, state, x_t, idx, device))
    return params


def _gaussian_forward_kl(base_params, cand_params, model) -> dict[str, float]:
    crop_chunk = int(model.config.action_chunk)
    crop_dim = int(model.config.action_env_dim)
    eps = 1e-8
    mean_vals = []
    sum_vals = []
    full_mean_vals = []

    for (mu0, std0), (mu1, std1) in zip(base_params, cand_params):
        var0 = std0.square().clamp_min(eps)
        var1 = std1.square().clamp_min(eps)
        kl = torch.log(std1.clamp_min(eps) / std0.clamp_min(eps))
        kl = kl + (var0 + (mu0 - mu1).square()) / (2.0 * var1) - 0.5
        selected = kl[:, :crop_chunk, :crop_dim]
        mean_vals.append(selected.mean())
        sum_vals.append(selected.sum(dim=(1, 2)).mean())
        full_mean_vals.append(kl.mean())

    return {
        "fkl_mean_per_dim": float(torch.stack(mean_vals).mean().item()),
        "fkl_sum_selected": float(torch.stack(sum_vals).mean().item()),
        "fkl_full_mean_per_dim": float(torch.stack(full_mean_vals).mean().item()),
    }


def _mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    keys = metrics_list[0].keys()
    return {
        key: sum(metrics[key] for metrics in metrics_list) / len(metrics_list)
        for key in keys
    }


def _build_base_cache(
    model,
    tasks: list[str],
    real_obs: dict[str, dict[str, Any]],
    batch_size: int,
    num_trajectories: int,
    seed_start: int,
    device: str,
):
    logger.info("=== Loading SFT base and sampling %d base SDE trajectories ===", num_trajectories)
    load_weights(model, MOE_SFT20K)
    model.eval()

    base_cache = {}
    for task in tasks:
        pfx, pkv, state = build_prefix_cache(model, real_obs[task], device)
        trajectories = []
        t_vals = None
        for traj_idx in range(num_trajectories):
            seed = seed_start + traj_idx
            x_traj, cur_t_vals, params = _sample_base_sde_trajectory(
                model,
                pfx,
                pkv,
                state,
                seed,
                batch_size,
                device,
            )
            t_vals = cur_t_vals
            trajectories.append({"seed": seed, "x_traj": x_traj, "params": params})
        base_cache[task] = {
            "t_vals": t_vals,
            "trajectories": trajectories,
        }
        logger.info("  base@%s done; trajectories=%d t=%s", task, len(trajectories), t_vals)
    return base_cache


def _eval_current_model_on_task(model, task: str, real_obs, base_cache, device: str):
    pfx, pkv, state = build_prefix_cache(model, real_obs[task], device)
    metrics_list = []
    for trajectory in base_cache[task]["trajectories"]:
        cand_params = _flow_sde_params(
            model,
            pfx,
            pkv,
            state,
            trajectory["x_traj"],
            device,
        )
        metrics_list.append(_gaussian_forward_kl(trajectory["params"], cand_params, model))
    return _mean_metrics(metrics_list)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["row_type"] == "candidate_task":
            grouped.setdefault(row["candidate"], []).append(row)

    summary = []
    for candidate, vals in grouped.items():
        ratios = [v["ratio_to_solo"] for v in vals if v["ratio_to_solo"] == v["ratio_to_solo"]]
        summary.append(
            {
                "candidate": candidate,
                "source": vals[0]["source"],
                "tasks": vals[0]["candidate_tasks"],
                "avg_fkl_mean_per_dim": sum(v["fkl_mean_per_dim"] for v in vals) / len(vals),
                "max_fkl_mean_per_dim": max(v["fkl_mean_per_dim"] for v in vals),
                "avg_ratio_to_solo": (sum(ratios) / len(ratios)) if ratios else float("nan"),
                "max_ratio_to_solo": max(ratios) if ratios else float("nan"),
            }
        )
    return sorted(summary, key=lambda x: (x["max_ratio_to_solo"], x["avg_ratio_to_solo"], x["candidate"]))


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (f"{v:.8f}" if isinstance(v, float) else v)
                    for k, v in row.items()
                }
            )


def _write_markdown(path: pathlib.Path, summary: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    num_trajectories = rows[0].get("num_trajectories", "unknown") if rows else "unknown"
    lines = [
        "# Flow-SDE Forward-KL Merge Scan",
        "",
        "Metric: analytic `KL(pi_base || pi_candidate)` over Flow-SDE Gaussian denoising transitions.",
        "The main value is averaged over base SDE trajectories, denoising steps, batch, action chunk, and env action dims.",
        f"Base SDE trajectories per task: `{num_trajectories}`.",
        "",
        "## Candidate Summary",
        "",
        "| candidate | source | tasks | avg KL/dim | max KL/dim | avg ratio | max ratio |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {candidate} | {source} | {tasks} | {avg_fkl_mean_per_dim:.6f} | "
            "{max_fkl_mean_per_dim:.6f} | {avg_ratio_to_solo:.3f} | {max_ratio_to_solo:.3f} |".format(**item)
        )

    lines.extend(
        [
            "",
            "## Per-Task Rows",
            "",
            "| candidate | task | KL/dim | KL/action | ratio_to_solo | solo_key |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        if row["row_type"] != "candidate_task":
            continue
        lines.append(
            "| {candidate} | {task} | {fkl_mean_per_dim:.6f} | {fkl_sum_selected:.4f} | "
            "{ratio_to_solo:.3f} | {solo_key} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _all_ppo_variants() -> dict[str, str]:
    variants = {f"old_{task}": path for task, path in OLD_BEST_PPO_WEIGHTS.items()}
    variants.update(STEP_PPO_WEIGHTS)
    return {k: v for k, v in variants.items() if pathlib.Path(v).exists()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/forward_kl_merge_sde20")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-trajectories", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--existing-only", action="store_true")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    candidates = _existing_candidates()
    if not args.existing_only:
        candidates.extend(_grid_candidates())

    needed_tasks = sorted({task for c in candidates for task in c.tasks})
    logger.info("Tasks needed: %s", needed_tasks)

    logger.info("=== Loading real observations ===")
    real_obs = {}
    for task in needed_tasks:
        real_obs[task] = load_real_obs_by_name(TASK_NAMES_EXT[task], args.batch_size)
        logger.info("  %s (%s): %r", task, TASK_NAMES_EXT[task], real_obs[task]["task_descriptions"][0])

    model = build_model(device)
    torch.manual_seed(args.seed_start)
    base_cache = _build_base_cache(
        model,
        needed_tasks,
        real_obs,
        args.batch_size,
        args.num_trajectories,
        args.seed_start,
        device,
    )

    rows: list[dict[str, Any]] = []
    solo_metrics: dict[tuple[str, str], dict[str, float]] = {}

    logger.info("=== Computing solo PPO forward-KL baselines ===")
    for key, weight_path in _all_ppo_variants().items():
        task = key.split("_", 1)[1] if key.startswith("old_") else "".join(ch for ch in key if not ch.isdigit())
        if task not in needed_tasks:
            continue
        load_weights(model, weight_path)
        model.eval()
        metrics = _eval_current_model_on_task(model, task, real_obs, base_cache, device)
        solo_metrics[(key, task)] = metrics
        rows.append(
            {
                "row_type": "solo",
                "candidate": key,
                "source": "solo_ppo",
                "candidate_tasks": task,
                "task": task,
                "solo_key": key,
                "fkl_mean_per_dim": metrics["fkl_mean_per_dim"],
                "fkl_sum_selected": metrics["fkl_sum_selected"],
                "fkl_full_mean_per_dim": metrics["fkl_full_mean_per_dim"],
                "ratio_to_solo": 1.0,
                "num_trajectories": args.num_trajectories,
            }
        )
        logger.info("  %s@%s: KL/dim %.6f", key, task, metrics["fkl_mean_per_dim"])

    base_sd = None
    ft_cache: dict[str, dict[str, torch.Tensor]] = {}
    if any(c.ft_keys for c in candidates):
        logger.info("=== Loading state dicts for in-memory grid candidates ===")
        base_sd = load_pt(MOE_SFT20K)
        all_ft_keys = sorted({key for c in candidates for key in c.ft_keys})
        ppo_variants = _all_ppo_variants()
        for key in all_ft_keys:
            ft_cache[key] = load_pt(ppo_variants[key])

    logger.info("=== Evaluating merge candidates ===")
    for candidate in candidates:
        logger.info("Candidate: %s", candidate.label)
        if candidate.path is not None:
            load_weights(model, candidate.path)
        else:
            assert base_sd is not None
            merged_sd = task_arithmetic(
                base_sd,
                [ft_cache[key] for key in candidate.ft_keys],
                alpha=list(candidate.alphas),
                include_prefixes=PI05_PPO_TRAINABLE_PREFIXES,
            )
            _validate_model_load(model, merged_sd, candidate.label)
            del merged_sd
        model.eval()

        for task in candidate.tasks:
            metrics = _eval_current_model_on_task(model, task, real_obs, base_cache, device)
            solo_key = (candidate.solo_keys or {}).get(task, "")
            solo = solo_metrics.get((solo_key, task))
            ratio = (
                metrics["fkl_mean_per_dim"] / solo["fkl_mean_per_dim"]
                if solo is not None and solo["fkl_mean_per_dim"] > 0
                else float("nan")
            )
            rows.append(
                {
                    "row_type": "candidate_task",
                    "candidate": candidate.label,
                    "source": candidate.source,
                    "candidate_tasks": "+".join(candidate.tasks),
                    "task": task,
                    "solo_key": solo_key,
                    "fkl_mean_per_dim": metrics["fkl_mean_per_dim"],
                    "fkl_sum_selected": metrics["fkl_sum_selected"],
                    "fkl_full_mean_per_dim": metrics["fkl_full_mean_per_dim"],
                    "ratio_to_solo": ratio,
                    "num_trajectories": args.num_trajectories,
                }
            )
            logger.info(
                "  %s@%s: KL/dim %.6f ratio %.3f",
                candidate.label,
                task,
                metrics["fkl_mean_per_dim"],
                ratio,
            )

    summary = _summarize(rows)
    _write_csv(out_dir / "forward_kl_rows.csv", rows)
    _write_csv(out_dir / "forward_kl_candidate_summary.csv", summary)
    _write_markdown(out_dir / "summary.md", summary, rows)
    logger.info("Wrote %s", out_dir.resolve())


if __name__ == "__main__":
    main()
