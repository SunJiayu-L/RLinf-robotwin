# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Label verified native-chunk data, validate every target, then train the critic."""

import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

REPO = Path("/home/tanhe/sjy/RLinf-robotwin")
ROOT = Path("/data2/sjy/recap/pick_dual_bottles/rollout2048_v2_chunk/critic_v2")


def status(stage: str, state: str, detail: str = "") -> None:
    """Publish training status independently of completed collection status."""
    event = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": stage,
        "status": state,
        "detail": detail,
    }
    path = ROOT / "pipeline_status.tmp"
    path.write_text(json.dumps(event, indent=2) + "\n")
    path.replace(ROOT / "pipeline_status.json")
    with (ROOT / "experiment_events.jsonl").open("a") as journal:
        journal.write(json.dumps(event) + "\n")


def validate_targets() -> dict:
    """Require exact MC rewards, frame keys, bounds and prompts for all episodes."""
    cfg = OmegaConf.load(ROOT / "protocol/returns.yaml")
    reports = []
    for entry in cfg.data.train_data_paths:
        directory = Path(entry.dataset_path)
        labels = pq.read_table(directory / "meta" / f"returns_{cfg.data.tag}.parquet")
        ep = labels["episode_index"].to_numpy()
        frame = labels["frame_index"].to_numpy()
        rewards = labels["reward"].to_numpy()
        returns = labels["return"].to_numpy()
        if not np.isfinite(returns).all() or returns.min() < -800 or returns.max() > 0:
            raise ValueError(f"Invalid return range: {directory}")
        if any(not text.strip() for text in labels["prompt"].to_pylist()):
            raise ValueError(f"Empty training prompt: {directory}")
        info = json.loads((directory / "meta/info.json").read_text())
        episode_records = [
            json.loads(line)
            for line in (directory / "meta/episodes.jsonl").read_text().splitlines()
        ]
        total = 0
        for index in range(info["total_episodes"]):
            episode_id = int(episode_records[index]["episode_index"])
            positions = np.flatnonzero(ep == episode_id)
            positions = positions[np.argsort(frame[positions])]
            if entry.type == "sft":
                length = int(episode_records[index]["length"])
                success = True
            else:
                relative = info["data_path"].format(
                    episode_index=episode_id,
                    episode_chunk=episode_id // info["chunks_size"],
                )
                source = pq.read_table(
                    directory / relative, columns=["frame_index", "is_success"]
                )
                np.testing.assert_array_equal(
                    frame[positions], source["frame_index"].to_numpy()
                )
                success = bool(np.asarray(source["is_success"][-1].as_py()).item())
                length = len(source)
            expected_rewards = np.full(length, -1, dtype=np.float32)
            expected_rewards[-1] = 0 if success else -400
            expected_returns = np.cumsum(expected_rewards[::-1])[::-1]
            np.testing.assert_array_equal(rewards[positions], expected_rewards)
            np.testing.assert_array_equal(returns[positions], expected_returns)
            total += length
        if total != len(labels):
            raise ValueError(f"Label row count mismatch: {directory}")
        reports.append(
            {
                "dataset": str(directory),
                "episodes": info["total_episodes"],
                "frames": total,
            }
        )
    report = {"datasets": reports, "status": "PASS", "tag": cfg.data.tag}
    (ROOT / "target_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def resource_status() -> dict:
    """Check available training resources without touching other users' jobs."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    devices = [list(map(int, line.split(","))) for line in result.stdout.splitlines()]
    memory = {
        line.split(":")[0]: int(line.split(":")[1].split()[0])
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    free_disk = shutil.disk_usage("/data2").free
    return {
        "ready": len(devices) == 8
        and all(free > 16000 and utilization < 30 for _, free, utilization in devices)
        and memory["MemAvailable"] > 160 * 1024 * 1024
        and free_disk > 300 * 1024**3,
        "gpu_index_free_mib_utilization": devices,
        "available_memory_kib": memory["MemAvailable"],
        "free_disk_bytes": free_disk,
    }


def wait_for_resources() -> bool:
    """Queue this experiment until all eight GPUs meet its launch requirements."""
    stable = 0
    while True:
        if (ROOT / "USER_PAUSED").exists():
            status("value", "PAUSED_BY_USER")
            return False
        resources = resource_status()
        stable = stable + 1 if resources["ready"] else 0
        if stable >= 2:
            return True
        status("value", "WAITING_FOR_RESOURCES", json.dumps(resources))
        time.sleep(30)


def main() -> None:
    """Run only return labeling and value training with final checkpoint checks."""
    with (ROOT / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for stage in ["returns", "value"]:
                if (ROOT / "USER_PAUSED").exists():
                    status(stage, "PAUSED_BY_USER")
                    return
                if (ROOT / f"{stage}.COMPLETE").exists():
                    continue
                if stage == "value":
                    preflight = json.loads(
                        (ROOT / "protocol/sampling_preflight.json").read_text()
                    )
                    if preflight["status"] != "PASS":
                        raise ValueError("Sampling preflight has not passed")
                    if not wait_for_resources():
                        return
                status(stage, "RUNNING")
                with (ROOT / f"logs/{stage}_launcher.log").open("a") as log:
                    result = subprocess.run(
                        ["bash", str(ROOT / "run_stage.sh"), stage],
                        cwd=REPO,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                if result.returncode:
                    raise RuntimeError(f"{stage} exited with {result.returncode}")
                if stage == "returns":
                    status(stage, "VALIDATING")
                    validate_targets()
                else:
                    actor = ROOT / "runs/value/checkpoints/global_step_8000/actor"
                    if not (actor / "dcp_checkpoint/.metadata").is_file():
                        raise FileNotFoundError(
                            "Final distributed critic checkpoint missing"
                        )
                    if len(list((actor / "dcp_checkpoint").glob("*.distcp"))) != 8:
                        raise ValueError(
                            "Final critic checkpoint has missing rank shards"
                        )
                    if not (actor / "model_state_dict/full_weights.pt").is_file():
                        raise FileNotFoundError("Final critic weights missing")
                (ROOT / f"{stage}.COMPLETE").write_text(
                    time.strftime("%Y-%m-%dT%H:%M:%S%z")
                )
            status("ALL", "COMPLETE", "8000 critic steps; no policy training")
        except Exception as error:
            status("ALL", "FAILED", repr(error))
            raise


if __name__ == "__main__":
    main()
