# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export ten verified failure videos per task from SFT40k."""

import fcntl
import json
import subprocess
import time
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path("/home/tanhe/sjy/RLinf-robotwin")
ROOT = Path("/data2/sjy/eval/robotwin6_SFT40k_failure_videos_20261106")
TASKS = [
    "beat_block_hammer",
    "place_phone_stand",
    "pick_dual_bottles",
    "lift_pot",
    "move_can_pot",
    "place_a2b_left",
]
CHECKPOINT = "/data2/sjy/checkpoints/robotwin8_A100_40k/full_weights.pt"


def status(task, state, detail=""):
    with (ROOT / "status.tsv").open("a") as stream:
        stream.write(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{task}\t{state}\t{detail}\n"
        )


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from examples.recap.export_failure_rollouts import export_recorded_hammer

    status(TASKS[0], "EXPORTING_RECORDED")
    export_recorded_hammer(ROOT / TASKS[0])
    status(TASKS[0], "DONE", "10 recorded failure videos")
    template = (
        REPO / "examples/recap/run_robotwin4096_sftpos_v4_restart.sh"
    ).read_text()
    for index, task in enumerate(TASKS[1:], 1):
        root = ROOT / task
        root.mkdir(exist_ok=True)
        (root / "logs").mkdir(exist_ok=True)
        if (root / "DONE.json").exists():
            continue
        if (root / "eval_policy").exists():
            raise RuntimeError(
                f"Partial output requires inspection before rerun: {root}"
            )
        seeds = [120_000_000 + index * 2_000_000 + i * 1024 for i in range(1024)]
        manifest = root / "scene_pool.json"
        manifest.write_text(json.dumps({task: {"success_seeds": seeds}}))
        cfg = OmegaConf.load(
            REPO / "examples/recap/config/robotwin_collect_4096_sftpos_v4_restart.yaml"
        )
        cfg.runner.logger.log_path = str(root / "runs")
        cfg.env.train.task_names = [task]
        cfg.env.train.task_config.task_name = task
        cfg.env.train.total_num_envs = 8
        cfg.env.train.seeds_path = str(manifest)
        cfg.env.train.task_config.save_path = str(root / "sim")
        cfg.env.train.episode_log_path = str(root / "logs/episodes.jsonl")
        cfg.actor.model.openpi.guidance_type = "no_guide"
        cfg.collect.num_envs_per_worker = 1
        cfg.collect.seed = 20261106
        cfg.collect.episodes_per_worker = 128
        cfg.collect.validation_episodes_per_worker = 0
        name = "robotwin_failure_video_" + task
        OmegaConf.save(cfg, REPO / "examples/recap/config" / f"{name}.yaml")
        shell = template.replace(
            "root=/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v4_restart",
            "root=" + str(root),
        )
        shell = shell.replace(
            "entry=examples/recap/collect_robotwin.py",
            "entry=examples/recap/export_failure_rollouts.py",
        )
        shell = shell.replace(
            "--config-name robotwin_collect_4096_sftpos_v4_restart",
            "--config-name " + name,
        )
        shell = shell.replace(
            "collect.episodes_per_worker=12", "collect.episodes_per_worker=128"
        )
        shell = shell.replace(
            '"collect.policy_checkpoint=$root/runs/policy/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"',
            '"collect.policy_checkpoint=' + CHECKPOINT + '"',
        )
        shell = shell.replace(
            "actor.model.openpi.guidance_type=positive",
            "actor.model.openpi.guidance_type=no_guide",
        )
        shell = shell.replace(
            " env.train.seeds_path=/data2/sjy/rl_protocol/robotwin6/eval_manifest_96ep.json",
            ' "env.train.seeds_path=' + str(manifest) + '"',
        )
        shell = shell.replace("7811", "7852").replace("/rt/r4v_", "/rt/fv6_")
        stage = root / "run_video.sh"
        stage.write_text(shell)
        subprocess.run(["bash", "-n", str(stage)], check=True)
        status(task, "RUNNING")
        with (root / "logs/launcher.log").open("a") as log:
            result = subprocess.run(
                ["bash", str(stage), "eval_policy"],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            status(task, "FAILED", str(result.returncode))
            raise RuntimeError(f"{task} failed: {result.returncode}")
        records = []
        for p in sorted((root / "eval_policy").glob("rank_*/failure_*.json")):
            row = json.loads(p.read_text())
            assert not row["success"] and row["task_name"] == task
            row["video"] = str(p.parent / row["video"])
            records.append(row)
        assert len(records) == 10, (task, len(records))
        assert len({x["seed"] for x in records}) == 10
        (root / "manifest.json").write_text(json.dumps(records, indent=2))
        (root / "DONE.json").write_text(json.dumps({"failure_videos": 10}))
        status(task, "DONE", "10 verified replay failure videos")
    (ROOT / "COMPLETE.json").write_text(json.dumps({"tasks": TASKS, "videos": 60}))
    status("ALL", "COMPLETE", "60 videos")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if ROOT.exists():
            status("SUPERVISOR", "FAILED", repr(exc))
        raise
