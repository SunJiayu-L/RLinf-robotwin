# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Finite RECAP experiment with heldout rollout checks and no automatic retry."""

import datetime
import fcntl
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

from validate_robotwin_rollouts import validate

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/data2/sjy/recap/beat_block_hammer/rollout4096_v2")
PYTHON = REPO / ".venv/bin/python"
ENTRY = REPO / "examples/recap/run_robotwin4096.sh"


def complete_checkpoint(experiment, step):
    actor = ROOT / "runs" / experiment / "checkpoints" / f"global_step_{step}" / "actor"
    if not (actor / "dcp_checkpoint/.metadata").is_file():
        return None
    shards = list((actor / "dcp_checkpoint").glob("*.distcp"))
    full = actor / "model_state_dict/full_weights.pt"
    if len(shards) != 8 or any(p.stat().st_size == 0 for p in shards):
        return None
    return actor if full.is_file() and full.stat().st_size > 100_000_000 else None


def qualified(metrics):
    """Engineering checks, independent of the legacy nine-dataset macro mean."""
    checks = {
        "rollout/continuous_mae": (0.15, "max"),
        "rollout/value_spearman": (0.3, "min"),
        "rollout/success_mae": (0.2, "max"),
        "rollout/failure_mae": (0.2, "max"),
        "rollout/value_spearman_defined": (1.0, "min"),
        "datasets_sft_val/continuous_mae": (0.05, "max"),
    }
    for key, (limit, direction) in checks.items():
        value = metrics.get(key, float("nan"))
        if not math.isfinite(value):
            return False
        if direction == "max" and value > limit:
            return False
        if direction == "min" and value < limit:
            return False
    return True


def save(state):
    state["updated_at"] = datetime.datetime.now().astimezone().isoformat()
    (ROOT / "pipeline_status.json").write_text(json.dumps(state, indent=2))


def run(stage, state, env=None):
    if (ROOT / "USER_PAUSED").exists():
        raise RuntimeError("User paused this experiment")
    state.update(stage=stage, status="RUNNING")
    save(state)
    with (ROOT / "logs" / (stage + "_launcher.log")).open("w") as stream:
        subprocess.run(
            ["bash", str(ENTRY), stage],
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    state["completed"].append(stage)
    save(state)


def main():
    state = {
        "status": "WAITING_PREFLIGHT",
        "completed": [],
        "started_at": datetime.datetime.now().astimezone().isoformat(),
    }
    with (ROOT / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(state)
        try:
            import sys

            if "--follow-collection" in sys.argv:
                existing = json.loads((ROOT / "collection_handoff.json").read_text())
                state.update(existing)
                state.update(status="RUNNING", stage="collect", supervisor_handoff=True)
                save(state)
                while True:
                    if (ROOT / "USER_PAUSED").exists():
                        raise RuntimeError("User paused this experiment")
                    rows = [
                        line.split("\t")
                        for line in (ROOT / "status.tsv").read_text().splitlines()
                    ]
                    matches = [
                        row for row in rows if len(row) > 2 and row[2] == "collect"
                    ]
                    if matches and matches[-1][1] == "END":
                        assert matches[-1][-1] == "exit=0", matches[-1]
                        break
                    time.sleep(10)
                with (ROOT / "stage.lock").open("a") as stage_lock:
                    fcntl.flock(stage_lock, fcntl.LOCK_EX)
                state["completed"].append("collect")
            else:
                deadline = time.monotonic() + 1800
                while True:
                    p = ROOT / "status.tsv"
                    rows = (
                        [line.split("\t") for line in p.read_text().splitlines()]
                        if p.exists()
                        else []
                    )
                    matches = [
                        row
                        for row in rows
                        if len(row) > 2 and row[2] == "collect_smoke"
                    ]
                    if matches and matches[-1][1] == "END":
                        assert matches[-1][-1] == "exit=0", matches[-1]
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("Collection smoke did not finish")
                    time.sleep(10)
                # The stage writes END before releasing its bounded Ray cleanup lock.
                with (ROOT / "stage.lock").open("a") as stage_lock:
                    fcntl.flock(stage_lock, fcntl.LOCK_EX)
                smoke = validate(ROOT / "datasets/rollout_smoke", 32)
                assert (
                    smoke["training_episodes"] == 16
                    and smoke["validation_episodes"] == 16
                )
                for p in (ROOT / "datasets/rollout_smoke").glob("rank_*/progress.json"):
                    assert json.loads(p.read_text())["saved_arrow_bytes"] == 0
                state["collection_smoke"] = smoke
                run("returns_smoke", state)
                run("value_smoke", state)
                assert complete_checkpoint("value_smoke", 2)
                rows = [
                    json.loads(line)
                    for line in (ROOT / "runs/value_smoke/value_validation.jsonl")
                    .read_text()
                    .splitlines()
                ]
                assert "rollout/value_spearman_defined" in rows[-1]["metrics"]
                paths = [
                    "examples/recap/collect_robotwin.py",
                    "examples/recap/run_robotwin4096.sh",
                    "examples/recap/run_robotwin4096_pipeline.py",
                    "rlinf/envs/robotwin/collection_utils.py",
                    "rlinf/algorithms/recap_value_metrics.py",
                    "rlinf/workers/sft/fsdp_value_sft_worker.py",
                ]
                paths += [
                    str(p.relative_to(REPO))
                    for p in (REPO / "examples/recap/config").glob(
                        "robotwin*4096*.yaml"
                    )
                ]
                (ROOT / "preflight_gate.json").write_text(
                    json.dumps(
                        {
                            "status": "PASS",
                            "collection": smoke,
                            "code_sha256": {
                                p: hashlib.sha256((REPO / p).read_bytes()).hexdigest()
                                for p in paths
                            },
                        },
                        indent=2,
                    )
                )
                run("collect", state)
            collected = validate(ROOT / "datasets/rollout", 4352)
            assert (
                collected["training_episodes"] == 4096
                and collected["validation_episodes"] == 256
            )
            state["collection"] = collected
            save(state)
            run("returns", state)
            run("value", state)
            assert complete_checkpoint("value", 8000)
            rows = [
                json.loads(line)
                for line in (ROOT / "runs/value/value_validation.jsonl")
                .read_text()
                .splitlines()
            ]
            candidates = [
                row
                for row in rows
                if qualified(row["metrics"])
                and complete_checkpoint("value", row["completed_steps"])
            ]
            if not candidates:
                state.update(
                    status="CRITIC_QUALITY_GATE_FAILED",
                    reason="No complete checkpoint passed heldout rollout and success/failure checks",
                )
                save(state)
                return
            selected = min(
                candidates, key=lambda row: row["metrics"]["rollout/continuous_mae"]
            )
            actor = complete_checkpoint("value", selected["completed_steps"])
            (ROOT / "selected_critic.json").write_text(
                json.dumps({"checkpoint": str(actor), **selected}, indent=2)
            )
            state["selected_critic_step"] = selected["completed_steps"]
            save(state)
            env = os.environ.copy()
            env["RECAP_VALUE_CHECKPOINT"] = str(actor)
            plan = (
                json.loads((ROOT / "execution_plan.json").read_text())
                if (ROOT / "execution_plan.json").exists()
                else {"stop_after": "value"}
            )
            if plan.get("stop_after") == "value":
                state.update(
                    status="PAUSED_AFTER_CRITIC",
                    reason="User requested no automatic policy training",
                )
                save(state)
                return
            run("adv", state, env=env)
            run("policy", state)
            assert complete_checkpoint("policy", 3000)
            for stage in ["eval_baseline", "eval_policy"]:
                run(stage, state)
                state[stage] = validate(ROOT / stage, 96)
                save(state)
            state["status"] = "COMPLETE"
            save(state)
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
