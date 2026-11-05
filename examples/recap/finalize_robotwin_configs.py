# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Finalize static recipes after the paper checks, before formal submission."""

from pathlib import Path

from omegaconf import OmegaConf

configs = Path(__file__).parent / "config"
for name in ["robotwin_adv_smoke", "robotwin_adv_iter000"]:
    path = configs / (name + ".yaml")
    cfg = OmegaConf.load(path)
    cfg.advantage.force_sft_positive = False
    cfg.advantage.threshold_reference_type = (
        None if name.endswith("smoke") else "rollout"
    )
    OmegaConf.save(cfg, path)
for name in ["robotwin_value_smoke", "robotwin_value_iter000"]:
    path = configs / (name + ".yaml")
    cfg = OmegaConf.load(path)
    for row in cfg.data.eval_data_paths:
        parent = Path(row.dataset_path).parent.name
        row["name"] = parent + "_" + Path(row.dataset_path).name
    OmegaConf.save(cfg, path)
path = configs / "robotwin_collect.yaml"
cfg = OmegaConf.load(path)
cfg.runner.logger.log_path = "/data2/sjy/recap/beat_block_hammer/iter000/runs"
cfg.runner.logger.experiment_name = "collection_iter000"
cfg.collect.policy_checkpoint = None
cfg.actor.model.openpi.guidance_type = "no_guide"
cfg.actor.model.openpi.positive_only_conditional = False
OmegaConf.save(cfg, path)
print("Finalized paper-aligned recipes")
