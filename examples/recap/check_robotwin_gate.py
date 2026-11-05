# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Record verifiable smoke results before permitting the formal RECAP iteration."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from run_robotwin_pipeline import checkpoint
from validate_robotwin_rollouts import validate

repo = Path(__file__).resolve().parents[2]
root = Path("/data2/sjy/recap/beat_block_hammer/iter000")
checkpoint("value_smoke", 2)
policy = checkpoint("policy_smoke", 2)
checkpoint("policy_batchcheck", 2)
checkpoint("value_batchcheck", 2)
rows = (root / "status.tsv").read_text().splitlines()
for stage in [
    "value_smoke",
    "cfg_smoke",
    "cfg_batchcheck",
    "value_batchcheck",
    "collect_smoke",
    "eval_smoke",
]:
    matches = [row for row in rows if "\tEND\t" + stage + "\t" in row]
    assert matches and matches[-1].endswith("exit=0"), (
        stage,
        matches[-1:] if matches else [],
    )
labels = pd.read_parquet(
    root / "datasets/sft_smoke/meta/advantages_robotwin_adv_smoke.parquet"
)
assert len(labels) == 912 and labels["advantage"].nunique() == 2
assert not labels[["episode_index", "frame_index"]].duplicated().any()
before = torch.load(
    "/data2/sjy/checkpoints/robotwin8_A100_40k/full_weights.pt",
    mmap=True,
    weights_only=True,
)
after = torch.load(policy, mmap=True, weights_only=True)
assert before.keys() == after.keys()
key = "action_out_proj.weight"
delta = (before[key].float() - after[key].float()).abs().max().item()
assert delta > 0 and torch.isfinite(after[key]).all(), delta
collection = validate(root / "datasets/rollout_smoke", 16)
evaluation = validate(root / "eval_smoke", 8)
paths = [
    "rlinf/data/datasets/recap/robotwin.py",
    "rlinf/data/datasets/recap/value_model.py",
    "rlinf/models/embodiment/value_model/checkpoint_utils.py",
    "rlinf/models/embodiment/value_model/modeling_critic.py",
    "rlinf/models/embodiment/openpi_cfg/checkpoint.py",
    "rlinf/models/embodiment/openpi_cfg/prompt_transforms.py",
    "rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py",
    "rlinf/models/embodiment/openpi_cfg/__init__.py",
    "rlinf/workers/sft/fsdp_cfg_worker.py",
    "rlinf/workers/sft/fsdp_value_sft_worker.py",
    "examples/recap/process/compute_returns.py",
    "examples/recap/process/compute_advantages.py",
    "examples/recap/collect_robotwin.py",
    "examples/recap/run_robotwin_recap.sh",
    "examples/recap/validate_robotwin_rollouts.py",
    "examples/recap/run_robotwin_pipeline.py",
]
paths += [
    str(path.relative_to(repo))
    for path in (repo / "examples/recap/config").glob("robotwin*.yaml")
]
gate = {
    "status": "PASS",
    "changed_action_projection_max_abs": delta,
    "advantage_frames": len(labels),
    "positive_labels": int(labels["advantage"].sum()),
    "collection": collection,
    "smoke_evaluation": evaluation,
    "code_sha256": {
        path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in paths
    },
}
(root / "smoke_gate.json").write_text(json.dumps(gate, indent=2))
print(json.dumps(gate))
