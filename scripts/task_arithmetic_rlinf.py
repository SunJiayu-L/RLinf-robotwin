#!/usr/bin/env python3
"""Task Arithmetic merging for RLinf full_weights.pt checkpoints.

公式:
    theta* = theta_SFT + alpha * sum_i (theta_RLi - theta_SFT)

多任务 (N=3) 时：
    theta* = theta_SFT + alpha * [(theta_adj - theta_SFT)
                                 + (theta_pcp - theta_SFT)
                                 + (theta_pec - theta_SFT)]

Usage:
    python scripts/task_arithmetic_rlinf.py \\
        --base   <sft_full_weights.pt> \\
        --ft     <adj.pt> <pcp.pt> <pec.pt> \\
        --output <merged_full_weights.pt> \\
        --alpha  0.35

参考: openpi/scripts/task_arithmetic_pt.py（safetensors 版本）
"""

from __future__ import annotations

import argparse
import logging
import pathlib

import torch

logger = logging.getLogger(__name__)

# pi0.5 PPO train_expert_only=True 下实际改变的参数白名单（208/812 keys）。
PI05_PPO_TRAINABLE_PREFIXES: tuple[str, ...] = (
    "paligemma_with_expert.gemma_expert.model.layers.",
    "paligemma_with_expert.gemma_expert.model.norm.",
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)


def load_pt(path: str | pathlib.Path) -> dict[str, torch.Tensor]:
    path = pathlib.Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(str(path))
    else:
        sd = torch.load(str(path), map_location="cpu")
        # 兼容被包在 {"state_dict": ...} 外层的格式
        if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
            sd = sd["state_dict"]
    logger.info(f"  loaded {len(sd)} tensors from {path}")
    return sd


def task_arithmetic(
    base_sd: dict[str, torch.Tensor],
    ft_sds: list[dict[str, torch.Tensor]],
    alpha: float | list[float],
    include_prefixes: tuple[str, ...] | None = PI05_PPO_TRAINABLE_PREFIXES,
) -> dict[str, torch.Tensor]:
    """theta* = theta_base + sum_i (alpha_i * tau_i)

    alpha can be a scalar (same for all tasks) or a list of per-task scalars.
    """
    n = len(ft_sds)
    if isinstance(alpha, (int, float)):
        alphas = [float(alpha)] * n
    else:
        if len(alpha) != n:
            raise ValueError(f"len(alpha)={len(alpha)} != len(ft_sds)={n}")
        alphas = [float(a) for a in alpha]

    eligible_keys = [
        k for k in base_sd
        if include_prefixes is None or any(k.startswith(p) for p in include_prefixes)
    ]
    logger.info(f"Eligible keys for merging: {len(eligible_keys)} / {len(base_sd)}")

    with torch.no_grad():
        # 合并：theta* = theta_base + sum_i (alpha_i * tau_i)
        merged: dict[str, torch.Tensor] = {}
        n_changed = 0
        for key, base_val in base_sd.items():
            if key in eligible_keys:  # type: ignore[operator]
                base_f32 = base_val.to(torch.float32)
                result = base_f32.clone()
                for ft, a in zip(ft_sds, alphas):
                    if key not in ft:
                        logger.warning(f"  skip (missing in ft): {key}")
                        continue
                    result = result + a * (ft[key].to(torch.float32) - base_f32)
                merged[key] = result.to(base_val.dtype)
                if not torch.equal(merged[key], base_val):
                    n_changed += 1
            else:
                merged[key] = base_val

    logger.info(f"Changed {n_changed} / {len(merged)} tensors vs base")
    return merged


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser(description="Task Arithmetic for RLinf full_weights.pt")
    ap.add_argument("--base", required=True, help="SFT base full_weights.pt path")
    ap.add_argument("--ft", required=True, nargs="+", help="PPO fine-tuned full_weights.pt path(s)")
    ap.add_argument("--output", required=True, help="Output full_weights.pt path")
    ap.add_argument("--alpha", type=float, default=0.35,
                    help="Uniform scaling coefficient for all task vectors (default 0.35)")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="Per-task alpha values (one per --ft path); overrides --alpha if given")
    ap.add_argument("--include-all", action="store_true",
                    help="Include all keys (not just PPO trainable prefixes)")
    args = ap.parse_args()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base: {args.base}")
    base_sd = load_pt(args.base)

    logger.info(f"Loading {len(args.ft)} fine-tuned checkpoint(s)...")
    ft_sds = [load_pt(p) for p in args.ft]

    alpha = args.alphas if args.alphas is not None else args.alpha
    include_prefixes = None if args.include_all else PI05_PPO_TRAINABLE_PREFIXES
    logger.info(f"alphas={alpha}, N={len(ft_sds)} tasks, preset={'all' if args.include_all else 'pi05_ppo'}")

    merged = task_arithmetic(base_sd, ft_sds, alpha=alpha, include_prefixes=include_prefixes)

    torch.save(merged, str(output_path))
    logger.info(f"Saved merged weights -> {output_path}")


if __name__ == "__main__":
    main()
