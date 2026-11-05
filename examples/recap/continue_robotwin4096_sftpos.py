# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Train a fresh expert-positive RECAP policy and evaluate one finite iteration."""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import run_robotwin4096_pipeline as pipeline
from omegaconf import OmegaConf
from validate_robotwin_rollouts import validate

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v3")
SOURCE = Path("/data2/sjy/recap/beat_block_hammer/rollout4096_v2")
pipeline.ROOT = ROOT
pipeline.ENTRY = REPO / "examples/recap/run_robotwin4096_sftpos.sh"


def preflight() -> dict:
    """Verify isolation, initialization, resources, and all label hashes."""
    assert (
        json.loads((SOURCE / "pipeline_status.json").read_text())["status"]
        == "PAUSED_BY_USER"
    )
    manifest = json.loads((ROOT / "expert_positive_manifest.json").read_text())
    assert (
        manifest["rollout_labels_unchanged"]
        and manifest["continuous_advantages_reused"]
    )
    for d in manifest["datasets"]:
        path = Path(d["path"])
        new = path / "meta/advantages_robotwin_adv_4096_sftpos_v3.parquet"
        old = path / "meta/advantages_robotwin_adv_4096_v2.parquet"
        assert hashlib.sha256(new.read_bytes()).hexdigest() == d["sha256"]
        assert hashlib.sha256(old.read_bytes()).hexdigest() == d["source_sha256"]
        if d["type"] == "sft":
            assert d["episodes"] == 450 and d["frames"] == d["positive_count"] == 51248
        else:
            assert d["sha256"] == d["source_sha256"]
    assert manifest["rollout_positive_fraction"] == 0.4
    cfg = OmegaConf.load(
        REPO / "examples/recap/config/robotwin_cfg_4096_sftpos_v3.yaml"
    )
    assert cfg.runner.resume_dir is None and cfg.runner.max_steps == 3000
    assert (
        str(cfg.actor.model.model_path) == "/data2/sjy/checkpoints/robotwin8_A100_40k"
    )
    assert cfg.actor.model.is_lora is False
    assert cfg.actor.model.openpi.train_expert_only is False
    assert cfg.actor.model.openpi.positive_only_conditional is False
    assert cfg.actor.model.openpi.unconditional_prob == 0.3
    assert cfg.data.advantage_tag == "robotwin_adv_4096_sftpos_v3"
    assert cfg.actor.micro_batch_size == 1 and cfg.actor.global_batch_size == 64
    assert cfg.actor.optim.lr == 5e-6 and cfg.actor.optim.lr_warmup_steps == 300
    assert str(cfg.runner.logger.log_path) == str(ROOT / "runs")
    for p in (ROOT / "runs/policy", ROOT / "eval_baseline", ROOT / "eval_policy"):
        assert not p.exists(), f"Existing output: {p}"
    assert not (ROOT / "USER_PAUSED").exists()
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    assert len(rows) == 8
    for row in rows:
        index, free, util = map(int, row.split(","))
        assert free > 16000 and util < 30, (index, free, util)
    mem = {
        l.split(":")[0]: int(l.split(":")[1].split()[0])
        for l in Path("/proc/meminfo").read_text().splitlines()
    }
    assert mem["MemAvailable"] > 160 * 1024 * 1024
    assert shutil.disk_usage("/data2").free > 300 * 1024**3
    return manifest


def main() -> None:
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = preflight()
        if "--preflight-only" in sys.argv:
            print(
                "PREFLIGHT_OK: SFT=100% positive; rollout unchanged=40%; SFT40k initialization; no resume"
            )
            return
        state = json.loads((ROOT / "pipeline_status.json").read_text())
        state["expected_mixed_positive_fraction"] = manifest[
            "expected_mixed_positive_fraction"
        ]
        state.update(
            started_at=datetime.datetime.now().astimezone().isoformat(),
            reference_critic_step=7000,
            policy_start="SFT40k, step0, no resume",
        )
        (ROOT / "execution_plan.json").write_text(
            json.dumps(
                {
                    "policy_training": True,
                    "stop_after": "eval_policy",
                    "max_steps": 3000,
                    "save_interval": 500,
                    "warmup": 300,
                    "global_batch_size": 64,
                    "micro_batch_size": 1,
                    "sft_positive_fraction": 1,
                    "rollout_positive_fraction": 0.4,
                    "expected_mixed_positive_fraction": 0.7,
                    "reference_critic_step": 7000,
                    "source_label_manifest": str(
                        ROOT / "expert_positive_manifest.json"
                    ),
                    "source_experiment_stopped": str(SOURCE),
                    "policy_initialization": "SFT40k, step0; no optimizer or policy state resume",
                    "stages": ["policy", "eval_baseline", "eval_policy"],
                    "runtime_check": "Reuse successful 2-step engine/checkpoint check; inspect early formal steps with corrected labels.",
                },
                indent=2,
            )
        )
        try:
            pipeline.run("policy", state)
            assert pipeline.complete_checkpoint("policy", 3000)
            for stage in ("eval_baseline", "eval_policy"):
                pipeline.run(stage, state)
                state[stage] = validate(ROOT / stage, 96)
                pipeline.save(state)
            state.update(status="COMPLETE", stage="eval_policy")
            pipeline.save(state)
            print("ITERATION_COMPLETE", json.dumps(state), flush=True)
        except Exception as error:
            state.update(
                status="PAUSED_BY_USER"
                if (ROOT / "USER_PAUSED").exists()
                else "FAILED",
                error=f"{type(error).__name__}: {error}",
            )
            pipeline.save(state)
            raise


if __name__ == "__main__":
    main()
