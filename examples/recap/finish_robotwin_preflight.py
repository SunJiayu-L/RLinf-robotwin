# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Finish bounded preflight stages, then execute the validated finite iteration."""

import fcntl
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/data2/sjy/recap/beat_block_hammer/iter000")
PYTHON = REPO / ".venv/bin/python"


def main():
    with (ROOT / "preflight.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 1200
        while True:
            rows = [
                line.split("\t")
                for line in (ROOT / "status.tsv").read_text().splitlines()
            ]
            matching = [
                row for row in rows if len(row) >= 3 and row[2] == "cfg_batchcheck"
            ]
            if matching and matching[-1][1] == "END":
                assert matching[-1][-1] == "exit=0", matching[-1]
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Policy batch preflight did not finish; no formal training started"
                )
            time.sleep(10)
        # Wait for the stage's bounded Ray cleanup before replacing its entry script.
        with (ROOT / "runner.lock").open("a") as stage_lock:
            fcntl.flock(stage_lock, fcntl.LOCK_EX)
            subprocess.run(
                [
                    "tar",
                    "-xzf",
                    "/data2/sjy/logs/recap_adaptation_20261030/recap_valuebatchcheck.tar.gz",
                    "-C",
                    str(REPO),
                ],
                check=True,
            )
        subprocess.run(
            [
                str(REPO / ".venv/bin/ruff"),
                "check",
                "--fix",
                "examples/recap/check_robotwin_gate.py",
            ],
            cwd=REPO,
            check=True,
        )
        subprocess.run(
            [
                str(REPO / ".venv/bin/ruff"),
                "format",
                "examples/recap/check_robotwin_gate.py",
            ],
            cwd=REPO,
            check=True,
        )
        for stage in ["value_batchcheck", "eval_smoke"]:
            if (ROOT / "USER_PAUSED").exists():
                raise RuntimeError("User paused RECAP preflight")
            print("Starting", stage, flush=True)
            with (ROOT / "logs" / (stage + "_launcher.log")).open("w") as stream:
                subprocess.run(
                    ["bash", str(REPO / "examples/recap/run_robotwin_recap.sh"), stage],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        subprocess.run(
            [str(PYTHON), str(REPO / "examples/recap/check_robotwin_gate.py")],
            check=True,
        )
        print("Preflight PASS; starting finite RECAP iteration", flush=True)
        with (ROOT / "logs/formal_pipeline_launcher.log").open("w") as stream:
            subprocess.run(
                [str(PYTHON), str(REPO / "examples/recap/run_robotwin_pipeline.py")],
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(json.dumps({"status": "COMPLETE"}), flush=True)


if __name__ == "__main__":
    main()
