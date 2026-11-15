# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collect 2048 fresh bottle-task episodes, validate, then train only the critic."""

import fcntl
import json
import subprocess
import time
from pathlib import Path

from omegaconf import OmegaConf

from examples.recap.prepare_robotwin import create_view
from examples.recap.validate_robotwin_rollouts import validate

REPO = Path("/home/tanhe/sjy/RLinf-robotwin")
ROOT = Path("/data2/sjy/recap/pick_dual_bottles/rollout2048_v1")
CONFIG = REPO / "examples/recap/config"
TASK = "pick_dual_bottles"
SUFFIX = "dual2048_v1"
TAG = "robotwin_time_dual2048_v1"


def entry(path, kind, weight=1.0, name=None):
    result = {
        "dataset_path": str(path),
        "type": kind,
        "weight": weight,
        "task_name": TASK,
        "robot_type": "robotwin",
        "model_type": "pi05",
        "return_min": -800.0,
        "return_max": 0.0,
    }
    if name:
        result["name"] = name
    return result


def status(stage, state, detail=""):
    with (ROOT / "status.tsv").open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\t{stage}\t{state}\t{detail}\n")
    (ROOT / "pipeline_status.json").write_text(
        json.dumps({"stage": stage, "status": state, "detail": detail}, indent=2)
    )


def prepare():
    if (ROOT / "protocol.json").exists():
        return
    for name in ["logs", "datasets", "protocol", "runs"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    source = Path(
        "/data2/sjy/datasets/lerobot/robotwin_demo_clean_aloha_agilex_23task_v21"
    )
    create_view(source, ROOT / "datasets/sft_train", list(range(4500, 4950)), TASK)
    create_view(source, ROOT / "datasets/sft_val", list(range(4950, 5000)), TASK)
    seeds = [160_000_000 + i * 1024 for i in range(2048)]
    (ROOT / "protocol/scene_pool.json").write_text(
        json.dumps({TASK: {"success_seeds": seeds}})
    )
    train = [entry(ROOT / "datasets/sft_train", "sft", 0.5)]
    train += [
        entry(ROOT / f"datasets/rollout/rank_{i:02d}/train", "rollout", 0.0625)
        for i in range(8)
    ]
    val = [entry(ROOT / "datasets/sft_val", "sft", name="sft_val")]
    val += [
        entry(
            ROOT / f"datasets/rollout/rank_{i:02d}/val",
            "rollout",
            name=f"rank_{i:02d}_val",
        )
        for i in range(8)
    ]
    value = OmegaConf.load(CONFIG / "robotwin_value_4096_v2.yaml")
    value.runner.logger.log_path = str(ROOT / "runs")
    value.data.train_data_paths = train
    value.data.eval_data_paths = val
    value.data.tag = TAG
    value.data.persistent_workers = False
    OmegaConf.save(value, CONFIG / f"robotwin_value_{SUFFIX}.yaml")
    returns = OmegaConf.load(CONFIG / "robotwin_returns_4096_v2.yaml")
    returns.data.train_data_paths = train + val
    returns.data.tag = TAG
    OmegaConf.save(returns, CONFIG / f"robotwin_returns_{SUFFIX}.yaml")
    collect = OmegaConf.load(CONFIG / "robotwin_collect_4096_v2.yaml")
    collect.runner.logger.log_path = str(ROOT / "runs")
    collect.env.train.task_names = [TASK]
    collect.env.train.task_config.task_name = TASK
    collect.env.train.total_num_envs = 16
    collect.env.train.seeds_path = str(ROOT / "protocol/scene_pool.json")
    collect.env.train.task_config.save_path = str(ROOT / "sim_collection")
    collect.env.train.episode_log_path = str(ROOT / "logs/collection_episodes.jsonl")
    collect.collect.output = str(ROOT / "datasets/rollout")
    collect.collect.seed = 20261106
    collect.collect.episodes_per_worker = 256
    collect.collect.validation_episodes_per_worker = 16
    collect.collect.num_envs_per_worker = 2
    OmegaConf.save(collect, CONFIG / f"robotwin_collect_{SUFFIX}.yaml")
    shell = (REPO / "examples/recap/run_robotwin4096_sftpos_v4_restart.sh").read_text()
    shell = shell.replace(
        "root=/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v4_restart",
        "root=" + str(ROOT),
    )
    shell = shell.replace("policy|eval_baseline|eval_policy)", "collect|returns|value)")
    shell = shell.replace(
        "robotwin_collect_4096_sftpos_v4_restart", f"robotwin_collect_{SUFFIX}"
    )
    shell = shell.replace("4096_v2", SUFFIX)
    shell = shell.replace("7811", "7891").replace("/rt/r4v_", "/rt/du2_")
    (ROOT / "run_stage.sh").write_text(shell)
    subprocess.run(["bash", "-n", str(ROOT / "run_stage.sh")], check=True)
    (ROOT / "protocol.json").write_text(
        json.dumps(
            {
                "task": TASK,
                "policy": "/data2/sjy/checkpoints/robotwin8_A100_40k/full_weights.pt",
                "rollout_total": 2048,
                "rollout_train": 1920,
                "rollout_val": 128,
                "sft_train": 450,
                "sft_val": 50,
                "critic_steps": 8000,
                "save_and_validate": 500,
                "global_batch": 128,
                "micro_batch": 4,
                "max_episode_steps": 400,
                "failure_reward": -400,
                "return_scale": 800,
                "critic_initialization": "pretrained vision/language backbones; no hammer critic resume",
                "persistent_workers": False,
                "policy_training": False,
            },
            indent=2,
        )
    )


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / "pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prepare()
    for stage in ["collect", "returns", "value"]:
        marker = ROOT / f"{stage}.COMPLETE"
        if marker.exists():
            continue
        if stage == "collect" and (ROOT / "datasets/rollout").exists():
            raise RuntimeError("Partial rollout exists; inspect before resubmitting")
        status(stage, "RUNNING")
        with (ROOT / f"logs/{stage}_launcher.log").open("a") as log:
            result = subprocess.run(
                ["bash", str(ROOT / "run_stage.sh"), stage],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"{stage} exited {result.returncode}")
        if stage == "collect":
            report = validate(ROOT / "datasets/rollout", 2048)
            assert (
                report["training_episodes"] == 1920
                and report["validation_episodes"] == 128
            )
            for p in (ROOT / "datasets/rollout").glob("rank_*/episodes.jsonl"):
                assert all(
                    json.loads(s)["task_name"] == TASK
                    for s in p.read_text().splitlines()
                )
        if stage == "value":
            actor = ROOT / "runs/value/checkpoints/global_step_8000/actor"
            assert (actor / "dcp_checkpoint/.metadata").is_file()
            assert len(list((actor / "dcp_checkpoint").glob("*.distcp"))) == 8
            assert (actor / "model_state_dict/full_weights.pt").stat().st_size > 0
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        status(stage, "COMPLETE")
    status(
        "ALL",
        "COMPLETE",
        "Rollout and critic finished; policy training was not launched",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("pipeline", "FAILED", repr(exc))
        raise
