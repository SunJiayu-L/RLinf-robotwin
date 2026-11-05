# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify complete trajectories and episode/seed separation before offline RL."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def validate(root: Path, expected: int) -> dict:
    records = []
    for rank in range(8):
        directory = root / f"rank_{rank:02d}"
        done = json.loads((directory / "DONE.json").read_text())
        rows = [
            json.loads(line)
            for line in (directory / "episodes.jsonl").read_text().splitlines()
        ]
        assert len(rows) == done["episodes"]
        for row in rows:
            assert row["instruction"].strip() and 0 < row["frames"] <= 400
            info = json.loads((directory / row["split"] / "meta/info.json").read_text())
            relative = info["data_path"].format(
                episode_index=row["episode_index"],
                episode_chunk=row["episode_index"] // info["chunks_size"],
            )
            table = pq.read_table(
                directory / row["split"] / relative,
                columns=["frame_index", "action", "observation.state", "is_success"],
            )
            assert table.num_rows == row["frames"]
            np.testing.assert_array_equal(
                table["frame_index"].to_numpy(), np.arange(row["frames"])
            )
            assert (
                bool(np.asarray(table["is_success"][-1].as_py()).item())
                == row["success"]
            )
            for key in ["action", "observation.state"]:
                values = np.stack(table[key].to_pylist())
                assert values.shape == (row["frames"], 14) and np.isfinite(values).all()
            records.append(row)
    assert len(records) == expected, (len(records), expected)
    seeds = [row["seed"] for row in records]
    assert len(seeds) == len(set(seeds)), (
        "Repeated rollout scene seed across train/validation shards"
    )
    if root.name.startswith("eval_"):
        manifest = json.loads(
            Path("/data2/sjy/rl_protocol/robotwin6/eval_manifest_96ep.json").read_text()
        )
        expected_pairs = {
            (row["seed"], row["instruction"])
            for row in manifest["beat_block_hammer"]["episodes"]
        }
        actual_pairs = {(row["seed"], row["instruction"]) for row in records}
        assert actual_pairs <= expected_pairs, (
            "Evaluation seed/instruction does not match held-out manifest"
        )
        if expected == len(expected_pairs):
            assert actual_pairs == expected_pairs, (
                "Evaluation omitted held-out episodes"
            )
    report = {
        "episodes": len(records),
        "frames": sum(row["frames"] for row in records),
        "successes": sum(row["success"] for row in records),
        "training_episodes": sum(row["split"] == "train" for row in records),
        "validation_episodes": sum(row["split"] == "val" for row in records),
        "all_seeds_unique": True,
    }
    (root / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.episodes)))
