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
    sd = torch.load(str(path), map_location="cpu")
    # 兼容被包在 {"state_dict": ...} 外层的格式
    if isinstance(sd, dict) and "state_dict" in sd and len(sd) == 1:
        sd = sd["state_dict"]
    logger.info(f"  loaded {len(sd)} tensors from {path}")
    return sd


def task_arithmetic(
    base_sd: dict[str, torch.Tensor],
    ft_sds: list[dict[str, torch.Tensor]],
    alpha: float,
    include_prefixes: tuple[str, ...] | None = PI05_PPO_TRAINABLE_PREFIXES,
) -> dict[str, torch.Tensor]:
    """theta* = theta_base + alpha * sum_i tau_i"""
    # 累积 task vector（fp32 精度计算，避免 bf16 累积误差）
    tv: dict[str, torch.Tensor] = {}
    eligible_keys = [
        k for k in base_sd
        if include_prefixes is None or any(k.startswith(p) for p in include_prefixes)
    ]
    logger.info(f"Eligible keys for merging: {len(eligible_keys)} / {len(base_sd)}")

    with torch.no_grad():
        for key in eligible_keys:
            base_val = base_sd[key].to(torch.float32)
            delta_sum = torch.zeros_like(base_val)
            for ft in ft_sds:
                if key not in ft:
                    logger.warning(f"  skip (missing in ft): {key}")
                    continue
                delta_sum += ft[key].to(torch.float32) - base_val
            tv[key] = delta_sum

        # 合并：theta* = theta_base + alpha * tau_sum
        merged: dict[str, torch.Tensor] = {}
        n_changed = 0
        for key, base_val in base_sd.items():
            if key in tv:
                result = base_val.to(torch.float32) + alpha * tv[key]
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
                    help="Scaling coefficient per task vector (default 0.35)")
    ap.add_argument("--include-all", action="store_true",
                    help="Include all keys (not just PPO trainable prefixes)")
    args = ap.parse_args()

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base: {args.base}")
    base_sd = load_pt(args.base)

    logger.info(f"Loading {len(args.ft)} fine-tuned checkpoint(s)...")
    ft_sds = [load_pt(p) for p in args.ft]

    include_prefixes = None if args.include_all else PI05_PPO_TRAINABLE_PREFIXES
    logger.info(f"alpha={args.alpha}, N={len(ft_sds)} tasks, preset={'all' if args.include_all else 'pi05_ppo'}")

    merged = task_arithmetic(base_sd, ft_sds, alpha=args.alpha, include_prefixes=include_prefixes)

    torch.save(merged, str(output_path))
    logger.info(f"Saved merged weights -> {output_path}")


if __name__ == "__main__":
    main()
