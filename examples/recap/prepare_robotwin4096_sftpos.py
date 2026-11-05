# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Version expert-positive labels without changing previous experiment data."""

from __future__ import annotations

import datetime
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
SOURCE = Path("/data2/sjy/recap/beat_block_hammer/rollout4096_v2")
ROOT = Path("/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v3")
OLD_TAG = "robotwin_adv_4096_v2"
TAG = "robotwin_adv_4096_sftpos_v3"


def sha(path: Path) -> str:
    """Return file SHA256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    assert (
        json.loads((SOURCE / "pipeline_status.json").read_text())["status"]
        == "PAUSED_BY_USER"
    )
    assert not ROOT.exists(), "Never overwrite an existing experiment"
    ROOT.mkdir(parents=True)
    for name in ("logs", "protocol", "runs"):
        (ROOT / name).mkdir()
    for suffix in ("cfg", "collect", "adv"):
        source = REPO / f"examples/recap/config/robotwin_{suffix}_4096_v2.yaml"
        cfg = OmegaConf.load(source)
        (ROOT / "protocol" / (source.name + ".before")).write_bytes(source.read_bytes())
        if suffix == "cfg":
            cfg.runner.logger.log_path = str(ROOT / "runs")
            cfg.runner.logger.experiment_name = "policy"
            cfg.runner.resume_dir = None
            cfg.data.advantage_tag = TAG
            cfg.actor.model.is_lora = False
        elif suffix == "collect":
            cfg = OmegaConf.create(
                OmegaConf.to_yaml(cfg).replace(str(SOURCE), str(ROOT))
            )
            cfg.env.train.episode_log_path = str(
                ROOT / "logs/collection_episodes.jsonl"
            )
        else:
            cfg.advantage.tag = TAG
            cfg.advantage.force_sft_positive = True
        OmegaConf.save(
            cfg, REPO / f"examples/recap/config/robotwin_{suffix}_4096_sftpos_v3.yaml"
        )
    cfg = OmegaConf.load(
        REPO / "examples/recap/config/robotwin_adv_4096_sftpos_v3.yaml"
    )
    original = json.loads((SOURCE / "advantage_report.json").read_text())
    original_by_path = {d["path"]: d for d in original["datasets"]}
    report = {
        "source_experiment": str(SOURCE),
        "tag": TAG,
        "source_tag": OLD_TAG,
        "critic_step": 7000,
        "threshold": original["threshold"],
        "expert_policy": "Confirmed successful expert demonstrations are positive behavioral anchors.",
        "continuous_advantages_reused": True,
        "rollout_labels_unchanged": True,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "datasets": [],
    }
    for entry in cfg.data.train_data_paths:
        path = Path(entry.dataset_path)
        old = path / f"meta/advantages_{OLD_TAG}.parquet"
        new = path / f"meta/advantages_{TAG}.parquet"
        assert not new.exists()
        assert sha(old) == original_by_path[str(path)]["sha256"]
        before = pd.read_parquet(old)
        assert np.isfinite(before.advantage_continuous).all()
        if entry.type == "sft":
            rewards = pd.read_parquet(
                path / "meta/returns_robotwin_time_4096_v2.parquet"
            )
            ends = (
                rewards.sort_values(["episode_index", "frame_index"])
                .groupby("episode_index")
                .tail(1)
            )
            assert len(ends) == 450 and ends.reward.eq(0).all()
            after = before.copy()
            after["advantage"] = True
            after.to_parquet(new, index=False)
            pd.testing.assert_frame_equal(
                before.drop(columns="advantage"),
                pd.read_parquet(new).drop(columns="advantage"),
            )
            assert pd.read_parquet(new).advantage.all()
        else:
            shutil.copy2(old, new)
            assert sha(new) == sha(old)
            after = before
        assert sha(old) == original_by_path[str(path)]["sha256"], "Old label modified"
        mixture_path = path / "meta/mixture_config.yaml"
        mixture = (
            yaml.safe_load(mixture_path.read_text()) if mixture_path.exists() else {}
        )
        mixture = mixture or {}
        tags = mixture.setdefault("tags", {})
        tags[TAG] = {
            **tags.get(OLD_TAG, {}),
            "source_tag": OLD_TAG,
            "force_sft_positive": True,
            "expert_positive_override": entry.type == "sft",
            "threshold_scope": "per_task",
            "threshold_reference_type": "rollout",
            "task_thresholds": {"beat_block_hammer": original["threshold"]},
        }
        mixture_path.write_text(yaml.safe_dump(mixture, sort_keys=False))
        report["datasets"].append(
            {
                "path": str(path),
                "type": entry.type,
                "frames": len(after),
                "episodes": int(after.episode_index.nunique()),
                "source_positive_count": int(before.advantage.sum()),
                "positive_count": int(after.advantage.sum()),
                "positive_fraction": float(after.advantage.mean()),
                "sha256": sha(new),
                "source_sha256": sha(old),
            }
        )
    sft = [r for r in report["datasets"] if r["type"] == "sft"]
    rollout = [r for r in report["datasets"] if r["type"] == "rollout"]
    assert (
        len(sft) == 1 and sft[0]["frames"] == 51248 and sft[0]["positive_fraction"] == 1
    )
    assert sum(r["episodes"] for r in rollout) == 4096
    report["rollout_positive_fraction"] = sum(
        r["positive_count"] for r in rollout
    ) / sum(r["frames"] for r in rollout)
    assert report["rollout_positive_fraction"] == 0.4
    report["expected_mixed_positive_fraction"] = (
        0.5 + 0.5 * report["rollout_positive_fraction"]
    )
    (ROOT / "expert_positive_manifest.json").write_text(json.dumps(report, indent=2))
    (ROOT / "pipeline_status.json").write_text(
        json.dumps(
            {
                "status": "READY",
                "stage": "labels_verified",
                "completed": ["labels_verified"],
                "source_experiment": str(SOURCE),
                "label_tag": TAG,
                "expected_mixed_positive_fraction": 0.7,
            },
            indent=2,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
