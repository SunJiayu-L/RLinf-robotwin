# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Reuse the completed GPU value estimates; correct SFT labels without inference."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from examples.recap.process.compute_advantages import save_advantages_to_dataset

root = Path("/data2/sjy/recap/beat_block_hammer/iter000/datasets/sft_smoke")
path = root / "meta/advantages_robotwin_adv_smoke.parquet"
frame = pd.read_parquet(path)
frame = frame.drop(columns="advantage").rename(
    columns={"advantage_continuous": "advantage"}
)
threshold = float(np.quantile(frame["advantage"], 0.6))
save_advantages_to_dataset(
    root,
    frame,
    threshold,
    dataset_type="sft",
    tag="robotwin_adv_smoke",
    force_sft_positive=False,
)
saved = pd.read_parquet(path)
assert saved["advantage"].nunique() == 2
assert len(saved) == len(saved[["episode_index", "frame_index"]].drop_duplicates())
print(
    json.dumps(
        {
            "frames": len(saved),
            "positive": int(saved["advantage"].sum()),
            "threshold": threshold,
        }
    )
)
