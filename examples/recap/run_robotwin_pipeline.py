# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Finite first RECAP iteration; stop on errors or an unreliable critic."""

import datetime
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from validate_robotwin_rollouts import validate

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/data2/sjy/recap/beat_block_hammer/iter000")
PYTHON = REPO / ".venv/bin/python"


def checkpoint(experiment: str, step: int):
    actor = ROOT / "runs" / experiment / "checkpoints" / f"global_step_{step}" / "actor"
    assert (actor / "dcp_checkpoint/.metadata").is_file()
    shards = list((actor / "dcp_checkpoint").glob("*.distcp"))
    assert len(shards) == 8 and all(p.stat().st_size > 0 for p in shards)
    weights = actor / "model_state_dict/full_weights.pt"
    assert weights.stat().st_size > 100_000_000
    return weights


def main():
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gate = json.loads((ROOT / "smoke_gate.json").read_text())
        assert gate["status"] == "PASS"
        for name, digest in gate["code_sha256"].items():
            assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest, (
                name
            )
        checkpoint("value_smoke", 2)
        checkpoint("policy_smoke", 2)
        stages = [
            "collect",
            "returns_iter000",
            "value_iter000",
            "adv_iter000",
            "cfg_iter000",
            "eval_baseline",
            "eval_iter000",
        ]
        state = {
            "started": datetime.datetime.now().astimezone().isoformat(),
            "completed": [],
            "status": "RUNNING",
        }
        status_path = ROOT / "pipeline_status.json"
        for stage in stages:
            if (ROOT / "USER_PAUSED").exists():
                raise RuntimeError("User paused this RECAP experiment")
            state["stage"] = stage
            status_path.write_text(json.dumps(state, indent=2))
            with (ROOT / "logs" / (stage + "_launcher.log")).open("w") as stream:
                code = subprocess.call(
                    ["bash", str(REPO / "examples/recap/run_robotwin_recap.sh"), stage],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            if code != 0:
                state.update(status="FAILED", exit_code=code)
                status_path.write_text(json.dumps(state, indent=2))
                sys.exit(code)
            if stage == "collect":
                state["collection"] = validate(ROOT / "datasets/rollout", 128)
            if stage == "value_iter000":
                checkpoint("value_iter000", 8000)
                last = json.loads(
                    (ROOT / "runs/value_iter000/value_validation.jsonl")
                    .read_text()
                    .splitlines()[-1]
                )
                metrics = last["metrics"]
                # Engineering acceptance criteria, not thresholds claimed by the paper.
                passed = (
                    metrics["continuous_mae"] <= 0.15
                    and metrics["value_spearman"] >= 0.3
                )
                state["critic_quality"] = {
                    "passed": passed,
                    "metrics": metrics,
                    "criteria": {"continuous_mae_max": 0.15, "spearman_min": 0.3},
                }
                if not passed:
                    state.update(status="CRITIC_QUALITY_GATE_FAILED")
                    status_path.write_text(json.dumps(state, indent=2))
                    sys.exit(8)
            if stage == "cfg_iter000":
                checkpoint("policy_iter000", 3000)
            if stage.startswith("eval_"):
                report = validate(ROOT / stage, 96)
                state[stage] = report
            state["completed"].append(stage)
            status_path.write_text(json.dumps(state, indent=2))
        state.update(
            status="COMPLETE", ended=datetime.datetime.now().astimezone().isoformat()
        )
        status_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status_path = ROOT / "pipeline_status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        status_path.write_text(json.dumps(state, indent=2))
        raise
