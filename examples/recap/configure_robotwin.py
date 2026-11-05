# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write reviewable, static RoboTwin RECAP recipes for the ZJU experiment."""

import json
from pathlib import Path

from omegaconf import OmegaConf


def main():
    repo = Path(__file__).resolve().parents[2]
    configs = repo / "examples/recap/config"
    configs.mkdir(exist_ok=True)
    root = Path("/data2/sjy/recap/beat_block_hammer/iter000")
    datasets = root / "datasets"
    source = json.loads(
        (datasets / "sft_train/meta/episodes.jsonl").read_text().splitlines()[0]
    )
    lengths = [
        json.loads(line)["length"]
        for split in ["sft_train", "sft_val"]
        for line in (datasets / split / "meta/episodes.jsonl").read_text().splitlines()
    ]
    failure_cost = max(400, max(lengths))
    bounds = {"return_min": -float(2 * failure_cost), "return_max": 0.0}
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "task": "beat_block_hammer",
                "failure_reward": -failure_cost,
                **bounds,
                "n_step": 50,
                "action_horizon": 50,
                "action_dim": 14,
                "fps": 15,
                "train_episodes": 450,
                "validation_episodes": 50,
                "first_source_episode": source,
                "positive_fraction": 0.4,
                "guidance_dropout": 0.3,
                "guidance_scale": 1.0,
                "initial_policy": "/data2/sjy/checkpoints/robotwin8_A100_40k/full_weights.pt",
                "paper": "https://arxiv.org/html/2511.14759v1",
            },
            indent=2,
        )
    )

    def entry(path, kind="sft", weight=1.0):
        return {
            "dataset_path": str(path),
            "type": kind,
            "weight": weight,
            "task_name": "beat_block_hammer",
            "robot_type": "robotwin",
            "model_type": "pi05",
            **bounds,
        }

    def write(name, cfg):
        target = configs / (name + ".yaml")
        if target.exists():
            raise FileExistsError(target)
        OmegaConf.save(cfg, target)

    cluster = {"num_nodes": 1, "component_placement": {"actor,env,rollout": "0-7"}}
    hydra = {
        "run": {"dir": "."},
        "output_subdir": None,
        "searchpath": ["file://${oc.env:REPO_PATH}/examples/sft/config/"],
    }
    for smoke in [True, False]:
        suffix = "smoke" if smoke else "iter000"
        train = [entry(datasets / ("sft_smoke" if smoke else "sft_train"))]
        if not smoke:
            train[0]["weight"] = 0.5
            train += [
                entry(
                    datasets / "rollout" / f"rank_{rank:02d}" / "train",
                    "rollout",
                    0.5 / 8,
                )
                for rank in range(8)
            ]
        val = [entry(datasets / "sft_val")]
        if not smoke:
            val += [
                entry(datasets / "rollout" / f"rank_{rank:02d}" / "val", "rollout")
                for rank in range(8)
            ]
        returns = OmegaConf.load(
            repo / "examples/recap/process/config/compute_returns.yaml"
        )
        returns.data.train_data_paths = train + val
        returns.data.failure_reward = -float(failure_cost)
        returns.data.tag = "robotwin_time_v1"
        returns.data.num_workers = 16
        write("robotwin_returns_" + suffix, returns)

        value = OmegaConf.load(
            repo / "examples/recap/value/config/libero_sft_value.yaml"
        )
        value.cluster = cluster
        value.hydra = hydra
        value.runner.logger.log_path = str(root / "runs")
        value.runner.logger.experiment_name = "value_" + suffix
        value.runner.max_steps = 2 if smoke else 8000
        value.runner.max_epochs = value.runner.max_steps
        value.runner.save_interval = 2 if smoke else 1000
        value.runner.val_check_interval = 2 if smoke else 200
        value.data.train_data_paths = train
        value.data.eval_data_paths = val
        value.data.tag = "robotwin_time_v1"
        value.data.update(
            {
                "action_dim": 14,
                "action_horizon": 50,
                "robot_type": "robotwin",
                "per_task_returns": True,
                **bounds,
                "balance_weights": False,
                "train_num_workers": 2,
                "eval_num_workers": 1,
                "prefetch_factor": 2,
            }
        )
        value.actor.micro_batch_size = 2 if smoke else 4
        value.actor.global_batch_size = 16 if smoke else 128
        value.actor.model.update(
            {
                "action_dim": 14,
                "action_horizon": 50,
                "critic_expert_variant": "gemma_100m",
                "siglip_path": "/data2/sjy/recap_assets/siglip2-so400m-patch14-224",
                "gemma3_path": "/data2/sjy/recap_assets/gemma-3-270m",
                "tokenizer_path": "/data2/sjy/recap_assets/gemma-3-270m",
            }
        )
        value.actor.fsdp_config.sharding_strategy = "full_shard"
        value.actor.optim.lr_warmup_steps = 1 if smoke else 500
        value.actor.optim.total_training_steps = value.runner.max_steps
        for row in value.data.eval_data_paths:
            row["max_samples"] = 16 if smoke else 1000
        write("robotwin_value_" + suffix, value)

        adv = OmegaConf.load(
            repo / "examples/recap/process/config/compute_advantages.yaml"
        )
        adv.defaults = ["_self_"]
        adv.data.train_data_paths = train
        adv.data.update(
            {
                "robot_type": "robotwin",
                "action_dim": 14,
                "advantage_lookahead_step": 50,
                **bounds,
            }
        )
        adv.advantage.value_checkpoint = str(
            root
            / "runs"
            / ("value_" + suffix)
            / "checkpoints"
            / ("global_step_2" if smoke else "global_step_8000")
            / "actor"
        )
        adv.advantage.positive_quantile = 0.4
        adv.advantage.threshold_scope = "per_task"
        adv.advantage.force_sft_positive = False
        adv.advantage.threshold_reference_type = None if smoke else "rollout"
        adv.advantage.tag = "robotwin_adv_" + suffix
        adv.advantage.returns_tag = "robotwin_time_v1"
        adv.advantage.batch_size = 8
        adv.advantage.num_dataloader_workers_per_gpu = 2
        adv.advantage.model = value.actor.model.copy()
        write("robotwin_adv_" + suffix, adv)

        cfg = OmegaConf.load(repo / "examples/recap/cfg/config/libero_cfg_openpi.yaml")
        cfg.cluster = cluster
        cfg.hydra = hydra
        cfg.runner.logger.log_path = str(root / "runs")
        cfg.runner.logger.experiment_name = "policy_" + suffix
        cfg.runner.max_steps = 2 if smoke else 3000
        cfg.runner.max_epochs = cfg.runner.max_steps
        cfg.runner.save_interval = 2 if smoke else 500
        cfg.data.train_data_paths = train
        cfg.data.advantage_tag = adv.advantage.tag
        cfg.data.balance_dataset_weights = False
        cfg.data.num_workers = 2
        cfg.actor.micro_batch_size = 1
        cfg.actor.global_batch_size = 8 if smoke else 64
        cfg.actor.model.update(
            {
                "model_path": "/data2/sjy/checkpoints/robotwin8_A100_40k",
                "strict_checkpoint": True,
                "num_action_chunks": 50,
                "action_dim": 14,
            }
        )
        cfg.actor.model.openpi.update(
            {
                "config_name": "pi05_aloha_robotwin",
                "action_chunk": 50,
                "action_env_dim": 14,
                "action_horizon": 50,
                "num_images_in_input": 3,
                "unconditional_prob": 0.3,
                "positive_only_conditional": False,
                "guidance_type": "positive",
                "cfgrl_guidance_scale": 1.0,
                "num_steps": 10,
            }
        )
        cfg.actor.fsdp_config.sharding_strategy = "full_shard"
        cfg.actor.fsdp_config.gradient_checkpointing = True
        cfg.actor.optim.lr_warmup_steps = 1 if smoke else 300
        cfg.actor.optim.total_training_steps = cfg.runner.max_steps
        cfg.actor.optim.lr = 5e-6
        write("robotwin_cfg_" + suffix, cfg)

    collect = OmegaConf.load(
        repo
        / "examples/embodiment/config/robotwin1_beat_block_hammer_A10040k_ppo200_w200_ep2_8gpu.yaml"
    )
    collect.defaults = [
        "env/robotwin_beat_block_hammer@env.train",
        "_self_",
        {"override hydra/job_logging": "stdout"},
    ]
    collect.hydra.searchpath = [
        "file://${oc.env:REPO_PATH}/examples/embodiment/config/"
    ]
    collect.cluster = {"num_nodes": 1, "component_placement": {"collector": "0-7"}}
    collect.actor.model.model_type = "cfg_model"
    collect.actor.model.add_value_head = False
    collect.actor.model.openpi.update(
        {
            "guidance_type": "no_guide",
            "positive_only_conditional": False,
            "action_chunk": 50,
            "action_env_dim": 14,
            "action_horizon": 50,
            "num_steps": 10,
        }
    )
    collect.actor.model.strict_checkpoint = True
    collect.env.train.total_num_envs = 16
    collect.env.train.auto_reset = False
    collect.env.train.ignore_terminations = False
    collect.env.train.episode_log_path = str(root / "logs/collection_episodes.jsonl")
    collect.env.train.task_config.save_path = str(root / "sim_collection")
    collect.collect = {
        "output": str(datasets / "rollout"),
        "seed": 20261031,
        "policy_checkpoint": None,
        "episodes_per_worker": 16,
        "validation_episodes_per_worker": 2,
        "num_envs_per_worker": 2,
        "fps": 15,
        "action_chunk": 50,
    }
    write("robotwin_collect", collect)
    print(configs)


if __name__ == "__main__":
    main()
