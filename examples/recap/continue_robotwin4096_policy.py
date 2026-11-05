# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Run one user-authorized RECAP policy iteration with the selected critic."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf
from run_robotwin4096_pipeline import REPO, ROOT, complete_checkpoint, run, save
from validate_robotwin_rollouts import validate


def preflight() -> dict:
    """Check inputs and absence of conflicting outputs without starting a job."""
    gpu_rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    assert len(gpu_rows) == 8
    for row in gpu_rows:
        index, free, util = map(int, row.split(","))
        assert free > 16000 and util < 30, (index, free, util)
    mem = {
        line.split(":")[0]: int(line.split(":")[1].split()[0])
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    assert mem["MemAvailable"] > 160 * 1024 * 1024
    assert shutil.disk_usage("/data2").free > 300 * 1024**3
    actor = complete_checkpoint("value", 7000)
    if actor is None:
        raise RuntimeError("Critic 7000 is not a complete checkpoint")
    rows = [
        json.loads(x)
        for x in (ROOT / "runs/value/value_validation.jsonl").read_text().splitlines()
    ]
    selected = next(row for row in rows if row["completed_steps"] == 7000)
    cfg = OmegaConf.load(REPO / "examples/recap/config/robotwin_cfg_4096_v2.yaml")
    assert cfg.runner.max_steps == 3000 and cfg.actor.global_batch_size == 64
    assert cfg.actor.micro_batch_size == 1 and cfg.runner.resume_dir is None
    assert cfg.actor.model.openpi.positive_only_conditional is False
    assert cfg.actor.model.openpi.unconditional_prob == 0.3
    assert cfg.actor.model.openpi.train_expert_only is False
    assert (
        str(cfg.actor.model.model_path) == "/data2/sjy/checkpoints/robotwin8_A100_40k"
    )
    assert cfg.actor.optim.lr == 5e-6 and cfg.actor.optim.lr_warmup_steps == 300
    adv = OmegaConf.load(REPO / "examples/recap/config/robotwin_adv_4096_v2.yaml")
    assert adv.advantage.force_sft_positive is False
    assert adv.advantage.threshold_reference_type == "rollout"
    assert adv.advantage.threshold_scope == "per_task"
    assert adv.advantage.positive_quantile == 0.4
    assert adv.data.advantage_lookahead_step == 50 and adv.data.gamma == 1
    for name in ("policy", "policy_smoke"):
        if (ROOT / "runs" / name).exists():
            raise RuntimeError(f"Existing {name} output; inspect before starting")
    for name in ("eval_baseline", "eval_policy"):
        if (ROOT / name).exists():
            raise RuntimeError(f"Existing {name} output; do not duplicate evaluation")
    for entry in adv.data.train_data_paths:
        path = Path(entry.dataset_path)
        assert "/val" not in str(path) and path.name != "sft_val"
        assert (path / "meta/returns_robotwin_time_4096_v2.parquet").is_file()
        if (path / "meta/advantages_robotwin_adv_4096_v2.parquet").exists():
            raise RuntimeError(f"Existing advantage output: {path}")
    assert not (ROOT / "USER_PAUSED").exists()
    paths = [
        "examples/recap/run_robotwin4096.sh",
        "examples/recap/config/robotwin_cfg_4096_v2.yaml",
        "examples/recap/config/robotwin_adv_4096_v2.yaml",
        "examples/recap/config/robotwin_collect_4096_v2.yaml",
        "examples/recap/continue_robotwin4096_policy.py",
    ]
    return dict(
        checkpoint=str(actor),
        **selected,
        selection="User accepted current critic; use lowest held-out rollout MAE checkpoint",
        legacy_gate_override="0.15 is an uncalibrated engineering gate, not a paper requirement",
        source_sha256={
            p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in paths
        },
    )


def validate_labels() -> dict:
    """Require exact frame coverage, finite values and consistent task labels."""
    import numpy as np
    import pandas as pd

    cfg = OmegaConf.load(REPO / "examples/recap/config/robotwin_adv_4096_v2.yaml")
    loaded = []
    for entry in cfg.data.train_data_paths:
        path = Path(entry.dataset_path)
        adv_path = path / "meta/advantages_robotwin_adv_4096_v2.parquet"
        adv = pd.read_parquet(adv_path).sort_values(["episode_index", "frame_index"])
        returns = pd.read_parquet(
            path / "meta/returns_robotwin_time_4096_v2.parquet"
        ).sort_values(["episode_index", "frame_index"])
        assert len(adv) == len(returns) > 0
        keys = ["episode_index", "frame_index"]
        assert not adv.duplicated(keys).any()
        assert np.array_equal(adv[keys].to_numpy(), returns[keys].to_numpy())
        assert np.isfinite(adv.advantage_continuous.to_numpy()).all()
        assert pd.api.types.is_bool_dtype(adv.advantage.dtype)
        assert adv.prompt.str.strip().str.len().gt(0).all() if "prompt" in adv else True
        loaded.append((entry.type, path, adv, adv_path))
    rollout = np.concatenate(
        [
            a.advantage_continuous.to_numpy()
            for kind, _, a, _ in loaded
            if kind == "rollout"
        ]
    )
    threshold = float(np.quantile(rollout, 0.6))
    for _, _, a, _ in loaded:
        assert np.array_equal(
            a.advantage.to_numpy(), a.advantage_continuous.to_numpy() >= threshold
        )
    report = {
        "task": "beat_block_hammer",
        "threshold": threshold,
        "threshold_reference": "training rollout frames",
        "positive_target": 0.4,
        "rollout_frames": len(rollout),
        "rollout_positive_fraction": float((rollout >= threshold).mean()),
        "datasets": [],
    }
    assert (rollout >= threshold).any() and (rollout < threshold).any()
    for kind, path, a, adv_path in loaded:
        report["datasets"].append(
            {
                "path": str(path),
                "type": kind,
                "frames": len(a),
                "episodes": int(a.episode_index.nunique()),
                "positive_fraction": float(a.advantage.mean()),
                "advantage_min": float(a.advantage_continuous.min()),
                "advantage_max": float(a.advantage_continuous.max()),
                "sha256": hashlib.sha256(adv_path.read_bytes()).hexdigest(),
            }
        )
    assert (
        sum(r["episodes"] for r in report["datasets"] if r["type"] == "rollout") == 4096
    )
    assert sum(r["episodes"] for r in report["datasets"] if r["type"] == "sft") == 450
    for path in (ROOT / "datasets/rollout").glob("rank_*/val"):
        assert not (path / "meta/advantages_robotwin_adv_4096_v2.parquet").exists()
    assert not (
        ROOT / "datasets/sft_val/meta/advantages_robotwin_adv_4096_v2.parquet"
    ).exists()
    (ROOT / "advantage_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selected = preflight()
        if "--preflight-only" in sys.argv:
            print(json.dumps(selected, indent=2))
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = ROOT / "protocol" / ("before_policy_" + stamp)
        backup.mkdir(parents=True, exist_ok=False)
        for name in ("pipeline_status.json", "execution_plan.json"):
            shutil.copy2(ROOT / name, backup / name)
        (ROOT / "selected_critic.json").write_text(json.dumps(selected, indent=2))
        (ROOT / "execution_plan.json").write_text(
            json.dumps(
                {
                    "stop_after": "eval_policy",
                    "policy_training": True,
                    "critic_step": 7000,
                    "user_authorized_at": datetime.datetime.now()
                    .astimezone()
                    .isoformat(),
                    "policy_initialization": "SFT40k, step0; no resume; full fine-tuning",
                    "stages": [
                        "adv",
                        "validate_labels",
                        "policy_smoke",
                        "policy",
                        "eval_baseline",
                        "eval_policy",
                    ],
                    "max_policy_steps": 3000,
                    "global_batch": 64,
                    "micro_batch": 1,
                    "warmup": 300,
                    "note": "User explicitly requested execution after accepting the current critic.",
                },
                indent=2,
            )
        )
        state = json.loads((ROOT / "pipeline_status.json").read_text())
        state.pop("reason", None)
        state.update(
            status="RUNNING",
            selected_critic_step=7000,
            critic_quality_gate_override=selected["legacy_gate_override"],
        )
        save(state)
        env = os.environ.copy()
        env["RECAP_VALUE_CHECKPOINT"] = selected["checkpoint"]
        try:
            run("adv", state, env=env)
            report = validate_labels()
            state["advantage_validation"] = report
            save(state)
            print("LABELS_VERIFIED", json.dumps(report), flush=True)
            run("policy_smoke", state)
            assert complete_checkpoint("policy_smoke", 2), (
                "Policy smoke checkpoint incomplete"
            )
            run("policy", state)
            assert complete_checkpoint("policy", 3000), (
                "Final policy checkpoint incomplete"
            )
            for stage in ("eval_baseline", "eval_policy"):
                run(stage, state)
                state[stage] = validate(ROOT / stage, 96)
                save(state)
            state.update(status="COMPLETE", stage="eval_policy")
            save(state)
            print("ITERATION_COMPLETE", json.dumps(state), flush=True)
        except Exception as error:
            state.update(
                status="PAUSED_BY_USER"
                if (ROOT / "USER_PAUSED").exists()
                else "FAILED",
                error=f"{type(error).__name__}: {error}",
            )
            save(state)
            raise


if __name__ == "__main__":
    main()
