# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the pick_dual_bottles MC policy labels and their dataset views."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

HORIZON = 50
TIMEOUT = 400
NEGATIVE_START = 300
REPO = Path(__file__).resolve().parents[2]


def rows(path: Path) -> list[dict]:
    """Read an episode journal."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def main() -> None:
    """Validate score coverage and publish isolated policy data views."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("critic_root", type=Path)
    parser.add_argument(
        "--threshold-percentile",
        type=int,
        choices=range(1, 100),
        metavar="1–99",
        default=60,
    )
    parser.add_argument(
        "--pool-sources",
        action="store_true",
        help="Use one advantage threshold and equal per-sample weights across sources.",
    )
    args = parser.parse_args()
    root = args.critic_root.resolve()
    quantile = args.threshold_percentile / 100
    version = (
        "v1" if args.threshold_percentile == 70 else f"p{args.threshold_percentile}"
    )
    if args.pool_sources:
        version += "_pooled"
    tag = f"pick_dual_mc3500_timeout300_{version}"
    scores = root / "analysis" / "mc_advantage_3500"
    dest = root / f"policy_mc3500_timeout300_{version}"
    if dest.exists():
        raise FileExistsError(dest)
    tables = {}
    for split in ("train", "val"):
        sources = [
            (
                f"rank_{rank:02d}/{split}",
                [scores / f"rollout_{split}_{rank:02d}.parquet"],
            )
            for rank in range(8)
        ]
        sources.append((f"sft_{split}", sorted(scores.glob(f"sft_{split}_*.parquet"))))
        for name, files in sources:
            source = root / "datasets" / name
            expert = name.startswith("sft_")
            df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
            df = df.sort_values(["episode_index", "frame_index"], ignore_index=True)
            lengths = {
                r["episode_index"]: r["length"]
                for r in rows(source / "meta/episodes.jsonl")
            }
            if df.groupby("episode_index").size().to_dict() != lengths:
                raise ValueError(f"Episode coverage mismatch for {name}")
            if not np.array_equal(
                df.frame_index, df.groupby("episode_index").cumcount()
            ):
                raise ValueError(f"Frame coverage mismatch for {name}")
            if not df.length.eq(df.episode_index.map(lengths)).all():
                raise ValueError(f"Frame lengths disagree with metadata for {name}")
            if expert:
                if not df.success.all():
                    raise ValueError(
                        f"Unexpected unsuccessful expert demonstration in {name}"
                    )
            else:
                journal = (
                    root.parent
                    / "datasets/rollout"
                    / name.split("/")[0]
                    / "episodes.jsonl"
                )
                journal_rows = {
                    r["episode_index"]: r for r in rows(journal) if r["split"] == split
                }
                if set(journal_rows) != set(lengths):
                    raise ValueError(f"Journal coverage mismatch for {name}")
                for ep, record in journal_rows.items():
                    if record["frames"] != lengths[ep] or (
                        not record["success"]
                        and (not record["truncated"] or lengths[ep] != TIMEOUT)
                    ):
                        raise ValueError(
                            f"Unexpected rollout termination in {name} episode {ep}"
                        )
                success = {ep: r["success"] for ep, r in journal_rows.items()}
                if not df.success.eq(df.episode_index.map(success)).all():
                    raise ValueError(f"Success flags disagree with journal for {name}")
            expected_returns = df.frame_index - df.length + 1 - 100 * (~df.success)
            if not np.array_equal(expected_returns, df.return_raw):
                raise ValueError(f"Unexpected reward convention in {name}")
            np.testing.assert_allclose(df.advantage, df.return_raw / 500 - df.value)
            if not np.isfinite(df.advantage).all():
                raise ValueError(f"Nonfinite advantages for {name}")
            tail = ~df.success & df.frame_index.ge(NEGATIVE_START)
            df["policy_eligible"] = ~tail | df.frame_index.add(HORIZON).le(df.length)
            df["forced_negative"] = tail & df.policy_eligible
            tables[name] = df

    reference = pd.concat(
        [
            df.loc[df.success | df.frame_index.lt(NEGATIVE_START), "advantage"]
            for name, df in tables.items()
            if name.endswith("/train") or (args.pool_sources and name == "sft_train")
        ],
        ignore_index=True,
    )
    threshold = float(reference.quantile(quantile))
    report = {
        "advantage_tag": tag,
        "critic_step": 3500,
        "failure_reward": -100,
        "return_scale": 500,
        "action_horizon": HORIZON,
        "timeout_steps": TIMEOUT,
        "forced_negative_start": NEGATIVE_START,
        "threshold_quantile": quantile,
        "threshold_reference_frames": len(reference),
        "threshold": threshold,
        "threshold_reference": "all_training_sources"
        if args.pool_sources
        else "rollout_only",
        "force_expert_positive": not args.pool_sources,
        "pool_sources": args.pool_sources,
        "datasets": {},
    }
    with tempfile.TemporaryDirectory(dir=root, prefix=".policy_labels_") as temp:
        staged = Path(temp) / "publish"
        staged.mkdir()
        for name, df in tables.items():
            source = root / "datasets" / name
            view = staged / "datasets" / name
            (view / "meta").mkdir(parents=True)
            for item in source.iterdir():
                if item.name != "meta":
                    (view / item.name).symlink_to(
                        item, target_is_directory=item.is_dir()
                    )
            for item in (source / "meta").iterdir():
                (view / "meta" / item.name).symlink_to(
                    item, target_is_directory=item.is_dir()
                )
            force_positive = name.startswith("sft_") and not args.pool_sources
            label = pd.Series(
                force_positive | (~df.forced_negative & df.advantage.gt(threshold)),
                dtype="boolean",
            )
            df["advantage_continuous"] = df["advantage"]
            df["advantage"] = label.where(df.policy_eligible)
            df.to_parquet(view / "meta" / f"advantages_{tag}.parquet", index=False)
            report["datasets"][name] = {
                "frames": len(df),
                "eligible": int(df.policy_eligible.sum()),
                "positive": int(df.advantage.sum()),
                "forced_negative": int(df.forced_negative.sum()),
                "excluded": int((~df.policy_eligible).sum()),
            }

        cfg = OmegaConf.load(
            REPO / "examples/recap/config/robotwin_cfg_4096_sftpos_v3.yaml"
        )
        cfg.runner.logger.log_path = str(dest / "runs")
        cfg.runner.logger.experiment_name = "policy"
        cfg.data.advantage_tag = tag
        if args.pool_sources:
            cfg.data.balance_dataset_weights = True
        for entry, name in zip(
            cfg.data.train_data_paths,
            ["sft_train", *[f"rank_{i:02d}/train" for i in range(8)]],
            strict=True,
        ):
            entry.dataset_path = str(dest / "datasets" / name)
            entry.task_name = "pick_dual_bottles"
            entry.return_min = -500.0
            if args.pool_sources:
                entry.weight = 1.0
        OmegaConf.save(cfg, staged / "robotwin_pick_dual_mc3500_timeout300.yaml")
        (staged / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        staged.rename(dest)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
