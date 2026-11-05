# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Create a separate 4096-train / 256-validation RoboTwin experiment."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
OLD = Path("/data2/sjy/recap/beat_block_hammer/iter000")
ROOT = Path("/data2/sjy/recap/beat_block_hammer/rollout4096_v2")
CONFIGS = REPO / "examples/recap/config"
TAG = "robotwin_time_4096_v2"


def entry(path, kind="sft", weight=1.0, name=None, maximum=None):
    result = {
        "dataset_path": str(path),
        "type": kind,
        "weight": weight,
        "task_name": "beat_block_hammer",
        "robot_type": "robotwin",
        "model_type": "pi05",
        "return_min": -800.0,
        "return_max": 0.0,
    }
    if name:
        result["name"] = name
    if maximum is not None:
        result["max_samples"] = maximum
    return result


def write(name, config):
    p = CONFIGS / (name + ".yaml")
    if p.exists():
        raise FileExistsError(p)
    OmegaConf.save(config, p)


def main():
    ROOT.mkdir(exist_ok=False)
    for folder in ["logs", "runs", "datasets", "protocol", "diagnostics"]:
        (ROOT / folder).mkdir()
    for split in ["sft_train", "sft_val", "sft_smoke"]:
        target = ROOT / "datasets" / split
        target.mkdir()
        shutil.copytree(
            OLD / "datasets" / split / "meta",
            target / "meta",
            ignore=shutil.ignore_patterns(
                "returns*.parquet", "advantages*.parquet", "mixture_config.yaml"
            ),
        )
        for name in ["data", "videos"]:
            source = OLD / "datasets" / split / name
            if source.exists():
                (target / name).symlink_to(source.resolve(), target_is_directory=True)
    # The existing RoboTwin interface calls this field success_seeds. Here they
    # are candidate scene seeds, never selected by the SFT policy's outcome.
    rng = np.random.default_rng(20261101)
    for name, count, base in [
        ("scene_pool", 4352, 1_000_000),
        ("smoke_pool", 32, 20_000_000),
    ]:
        seeds = (base + 1024 * np.arange(count)).tolist()
        rng.shuffle(seeds)
        (ROOT / "protocol" / (name + ".json")).write_text(
            json.dumps({"beat_block_hammer": {"success_seeds": seeds}})
        )
    norm = Path(
        "/data2/sjy/checkpoints/robotwin8_A100_40k/physical-intelligence/robotwin/norm_stats.json"
    )
    protocol = {
        "task": "beat_block_hammer",
        "training_rollouts": 4096,
        "validation_rollouts": 256,
        "total_collection": 4352,
        "sft_train": 450,
        "sft_validation": 50,
        "initial_policy": "/data2/sjy/checkpoints/robotwin8_A100_40k/full_weights.pt",
        "seed_pool_semantics": "candidate scenes, not filtered by policy success; actual simulator seed recorded",
        "seed_spacing": 1024,
        "normalization_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
        "failure_reward": -400,
        "return_min": -800.0,
        "return_max": 0.0,
        "n_step": 50,
        "critic_steps": 8000,
        "policy_steps": 3000,
        "gpu_count": 8,
        "ray_cpus": 64,
        "minimum_available_ram_gib": 100,
        "saved_arrow_cache_bytes": 0,
        "selection": "best complete 500-step critic checkpoint passing heldout rollout and success/failure checks",
        "quality_criteria_are_engineering_choices": True,
    }
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for smoke in [False, True]:
        suffix = "4096_smoke" if smoke else "4096_v2"
        rollout = "rollout_smoke" if smoke else "rollout"
        train = [
            entry(
                ROOT / "datasets" / ("sft_smoke" if smoke else "sft_train"), weight=0.5
            )
        ]
        train += [
            entry(
                ROOT / "datasets" / rollout / f"rank_{rank:02d}" / "train",
                "rollout",
                0.0625,
            )
            for rank in range(8)
        ]
        val = [
            entry(
                ROOT / "datasets" / "sft_val",
                name="datasets_sft_val",
                maximum=16 if smoke else None,
            )
        ]
        val += [
            entry(
                ROOT / "datasets" / rollout / f"rank_{rank:02d}" / "val",
                "rollout",
                name=f"rank_{rank:02d}_val",
                maximum=16 if smoke else None,
            )
            for rank in range(8)
        ]
        returns = OmegaConf.load(CONFIGS / "robotwin_returns_iter000.yaml")
        returns.data.train_data_paths = train + val
        returns.data.tag = TAG
        write("robotwin_returns_" + suffix, returns)
        value = OmegaConf.load(CONFIGS / "robotwin_value_iter000.yaml")
        value.runner.logger.log_path = str(ROOT / "runs")
        value.runner.logger.experiment_name = "value_smoke" if smoke else "value"
        value.runner.max_steps = 2 if smoke else 8000
        value.runner.max_epochs = value.runner.max_steps
        value.runner.save_interval = 2 if smoke else 500
        value.runner.val_check_interval = 2 if smoke else 500
        value.runner.resume_dir = None
        value.data.train_data_paths = train
        value.data.eval_data_paths = val
        value.data.tag = TAG
        value.actor.optim.lr_warmup_steps = 1 if smoke else 500
        value.actor.optim.total_training_steps = value.runner.max_steps
        write("robotwin_value_" + suffix, value)
        if not smoke:
            adv = OmegaConf.load(CONFIGS / "robotwin_adv_iter000.yaml")
            adv.data.train_data_paths = train
            adv.advantage.value_checkpoint = str(
                ROOT / "runs/value/checkpoints/global_step_8000/actor"
            )
            adv.advantage.returns_tag = TAG
            adv.advantage.tag = "robotwin_adv_4096_v2"
            write("robotwin_adv_4096_v2", adv)
            policy = OmegaConf.load(CONFIGS / "robotwin_cfg_iter000.yaml")
            policy.runner.logger.log_path = str(ROOT / "runs")
            policy.runner.logger.experiment_name = "policy"
            policy.runner.resume_dir = None
            policy.data.train_data_paths = train
            policy.data.advantage_tag = adv.advantage.tag
            write("robotwin_cfg_4096_v2", policy)
    collect = OmegaConf.load(CONFIGS / "robotwin_collect.yaml")
    collect.runner.logger.log_path = str(ROOT / "runs")
    collect.runner.logger.experiment_name = "collection"
    collect.env.train.seeds_path = str(ROOT / "protocol/scene_pool.json")
    collect.env.train.task_config.save_path = str(ROOT / "sim_collection")
    collect.collect.output = str(ROOT / "datasets/rollout")
    collect.collect.episodes_per_worker = 544
    collect.collect.validation_episodes_per_worker = 32
    collect.collect.num_envs_per_worker = 2
    collect.collect.seed = 20261101
    write("robotwin_collect_4096_v2", collect)
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
