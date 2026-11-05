# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Read actual video-backed train/heldout samples through the critic pipeline."""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from rlinf.data.datasets.recap.value_model import ValueDataset
from rlinf.models.embodiment.value_model.data_collator import ValueDataCollator
from rlinf.models.embodiment.value_model.processing import ValueProcessor

root = Path("/data2/sjy/recap/beat_block_hammer/iter000")
protocol = json.loads((root / "protocol.json").read_text())
results = {}
tokenizer = AutoTokenizer.from_pretrained(
    "/data2/sjy/recap_assets/gemma-3-270m", local_files_only=True
)
collator = ValueDataCollator(processor=ValueProcessor(tokenizer=tokenizer), train=False)
for split in ["sft_smoke", "sft_val"]:
    dataset = ValueDataset(
        str(root / "datasets" / split),
        robot_type="robotwin",
        model_type="pi05",
        action_dim=14,
        action_horizon=50,
        return_min=protocol["return_min"],
        return_max=0.0,
        tag="robotwin_time_v1",
    )
    samples = [dataset[0], dataset[len(dataset) - 1]]
    batch = collator(samples)
    assert np.isfinite([row["target_values"] for row in samples]).all()
    results[split] = {
        "frames": len(dataset),
        "prompts": [row["prompt"] for row in samples],
        "targets": [row["target_values"] for row in samples],
        "image_keys": list(samples[0]["images"]),
        "batch_images": {
            key: list(value.shape)
            for key, value in batch["observation"]["images"].items()
        },
    }
print(json.dumps(results, indent=2))
