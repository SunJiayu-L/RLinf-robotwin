# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Validate chunk-recording parity, collect 2048 episodes, then stop."""

import argparse
import fcntl
import hashlib
import io
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from examples.recap.collection_resume import prepare_resume, read_summaries
from examples.recap.validate_robotwin_rollouts import validate

REPO = Path("/home/tanhe/sjy/RLinf-robotwin")
ROOT = Path("/data2/sjy/recap/pick_dual_bottles/rollout2048_v2_chunk")
REFERENCE = Path(
    "/data2/sjy/eval/pick_dual_bottles_sft40k_20261107/paired_chunk/chunk10"
)


def write_status(stage: str, state: str, detail: str = "") -> None:
    """Publish one complete status record."""
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": stage,
        "status": state,
        "detail": detail,
    }
    temporary = ROOT / "pipeline_status.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(ROOT / "pipeline_status.json")


def check_smoke() -> dict:
    """Require all 96 recorded runs to match their unrecorded native reference."""
    report = validate(ROOT / "smoke96", 96)
    checks = []
    for rank in range(8):
        directory = ROOT / "smoke96" / f"rank_{rank:02d}"
        baseline = REFERENCE / f"rank_{rank:02d}"
        actual = [
            json.loads(s)
            for s in (directory / "episodes.jsonl").read_text().splitlines()
        ]
        expected = [
            json.loads(s)
            for s in (baseline / "episodes.jsonl").read_text().splitlines()
        ]
        for row, ref in zip(actual, expected, strict=True):
            for field in ["seed", "instruction", "policy_noise_seed", "success"]:
                if row[field] != ref[field]:
                    raise ValueError(
                        f"Recording changed {field} at scene {row['seed']}"
                    )
            table = pq.read_table(
                directory
                / row["split"]
                / "data/chunk-000"
                / f"episode_{row['episode_index']:06d}.parquet",
                columns=[
                    "action",
                    "observation.state",
                    "observation.images.cam_high",
                    "observation.images.cam_left_wrist",
                    "observation.images.cam_right_wrist",
                ],
            )
            first = table.slice(0, 1).to_pylist()[0]
            images = [
                np.asarray(
                    Image.open(
                        io.BytesIO(first[f"observation.images.{camera}"]["bytes"])
                    ).convert("RGB")
                )
                for camera in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
            ]
            arrays = {
                "main_images": np.stack([images[0]]),
                "wrist_images": np.stack([images[1:]]),
                "states": np.asarray([first["observation.state"]], dtype=np.float32),
            }
            hashes = {
                name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in arrays.items()
            }
            if hashes != ref["initial_hashes"]:
                raise ValueError(f"Initial observation changed at scene {row['seed']}")
            count = min(50, len(table))
            recorded = np.asarray(
                table["action"].slice(0, count).to_pylist(), dtype=np.float32
            )
            original = np.load(
                baseline / f"initial_actions_{row['episode_index']:02d}.npy"
            )[0, :count]
            difference = float(np.abs(recorded - original).max())
            if difference > 1e-6:
                raise ValueError(
                    f"Initial actions changed: seed={row['seed']}, delta={difference}"
                )
            checks.append(
                {
                    "seed": row["seed"],
                    "success": row["success"],
                    "action_max_abs_diff": difference,
                }
            )
    report.update(
        {
            "matched_reference_episodes": len(checks),
            "max_initial_action_difference": max(
                row["action_max_abs_diff"] for row in checks
            ),
            "checks": checks,
        }
    )
    (ROOT / "smoke_parity.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def collect_segments(*, resuming: bool) -> dict:
    """Recycle simulator processes and append only missing committed episodes."""
    destination = ROOT / "datasets/rollout"
    if destination.exists() and not resuming:
        raise FileExistsError("Refusing to overwrite existing collection")
    no_progress_failures = 0
    for attempt in range(1, 25):
        if (ROOT / "USER_PAUSED").exists():
            write_status("collect", "PAUSED_BY_USER")
            return {}
        before = 0
        recovery = {}
        for rank in range(8):
            directory = destination / f"rank_{rank:02d}"
            if directory.exists():
                recovery[directory.name] = prepare_resume(directory, 256, 16)
                before += recovery[directory.name]["episodes"]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (ROOT / f"recovery_{stamp}.json").write_text(
            json.dumps(recovery, indent=2) + "\n"
        )
        if before == 2048:
            break
        # First prove checkpoint seeking and partial-batch replay on a short run.
        limit = 1 if resuming and attempt == 1 else 12
        write_status(
            "collect",
            "RUNNING",
            f"{before}/2048 committed; segment {stamp}; at most {limit} batches per worker",
        )
        log = ROOT / f"logs/collect_launcher_{stamp}.log"
        environment = os.environ.copy()
        environment["RLINF_COLLECT_BATCH_LIMIT"] = str(limit)
        environment["RLINF_COLLECT_LOG_SUFFIX"] = f"_{stamp}"
        with log.open("w") as stream:
            process = subprocess.Popen(
                ["bash", str(ROOT / "run_stage.sh"), "collect"],
                cwd=REPO,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            (ROOT / "collect_launch.json").write_text(
                json.dumps({"pid": process.pid, "segment": stamp, "log": str(log)})
                + "\n"
            )
            code = process.wait()
        after = sum(
            len(read_summaries(destination / f"rank_{rank:02d}", 256, 16))
            for rank in range(8)
        )
        event = {"segment": stamp, "before": before, "after": after, "exit_code": code}
        with (ROOT / "segments.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if code:
            worker_log = ROOT / f"logs/collect_{stamp}.log"
            error = worker_log.read_text() if worker_log.exists() else log.read_text()
            if "cannot create buffer" not in error:
                raise RuntimeError(
                    f"Collection segment failed with code {code}; see {log}"
                )
            no_progress_failures = no_progress_failures + 1 if after <= before else 0
            if no_progress_failures >= 3:
                raise RuntimeError(
                    "Rendering failed on three attempts without progress"
                )
        elif after <= before:
            raise RuntimeError(
                "Collection segment returned without saving new episodes"
            )
        else:
            no_progress_failures = 0
    else:
        raise RuntimeError("Exceeded the bounded collection restart count")
    write_status("validate", "RUNNING")
    return validate(destination, 2048)


def main(
    *, validate_existing_smoke: bool = False, resume_collection: bool = False
) -> None:
    """Run collection only after the recording implementation passes live parity."""
    with (ROOT / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if resume_collection:
                report = json.loads((ROOT / "smoke.COMPLETE").read_text())
                if report["matched_reference_episodes"] != 96:
                    raise ValueError(
                        "Cannot resume without completed recording parity checks"
                    )
            else:
                write_status(
                    "smoke", "RUNNING", "Waiting for 96 native-chunk recording checks"
                )
                pid = json.loads((ROOT / "smoke_launch.json").read_text())["pid"]
                while Path(f"/proc/{pid}/stat").exists():
                    if (
                        Path(f"/proc/{pid}/stat")
                        .read_text()
                        .split(") ", 1)[1]
                        .startswith("Z")
                    ):
                        break
                    time.sleep(15)
                if (
                    not validate_existing_smoke
                    and "\tEND\tsmoke\texit=0" not in (ROOT / "status.tsv").read_text()
                ):
                    raise RuntimeError("Smoke collection did not exit successfully")
                write_status(
                    "smoke", "VALIDATING", "Checking all 96 saved trajectories"
                )
                report = check_smoke()
                (ROOT / "smoke.COMPLETE").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            if (ROOT / "USER_PAUSED").exists():
                write_status("collect", "PAUSED_BY_USER")
                return
            result = collect_segments(resuming=resume_collection)
            if not result:
                return
            if (
                result["training_episodes"] != 1920
                or result["validation_episodes"] != 128
            ):
                raise ValueError("Unexpected train/validation split")
            (ROOT / "collect.COMPLETE").write_text(json.dumps(result, indent=2) + "\n")
            write_status(
                "ALL",
                "COMPLETE",
                f"{result['successes']}/2048 successes; collection only",
            )
        except Exception as error:
            write_status("ALL", "FAILED", repr(error))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume-collection",
        action="store_true",
        help="Preserve committed episodes and resume in bounded simulator processes.",
    )
    parser.add_argument(
        "--validate-existing-smoke",
        action="store_true",
        help=(
            "Recover from a launcher-only failure by fully validating all saved "
            "smoke trajectories and reference parity before starting collection."
        ),
    )
    arguments = parser.parse_args()
    main(
        validate_existing_smoke=arguments.validate_existing_smoke,
        resume_collection=arguments.resume_collection,
    )
