# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify complete trajectories and episode/seed separation before offline RL."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def validate(root: Path, expected: int, *, require_complete: bool = True) -> dict:
    """Validate committed episodes, optionally auditing an interrupted collection."""
    records = []
    for rank in range(8):
        directory = root / f"rank_{rank:02d}"
        rows = [
            json.loads(line)
            for line in (directory / "episodes.jsonl").read_text().splitlines()
        ]
        if require_complete:
            done = json.loads((directory / "DONE.json").read_text())
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
            if row.get("execution_mode") == "native_full_chunk_topp":
                counts = row["executed_chunk_lengths"]
                assert sum(counts) == row["frames"]
                assert all(0 < count <= 50 for count in counts)
                assert all(count == 50 for count in counts[:-1])
                assert row["success"] or row["truncated"]
                if not row["success"]:
                    assert row["frames"] == 400 and row["truncated"]
                with np.load(directory / row["terminal_observation"]) as terminal:
                    assert terminal["state"].shape == (14,)
                    assert np.isfinite(terminal["state"]).all()
                    assert terminal["images"].shape == (3, 240, 320, 3)
                    assert terminal["images"].dtype == np.uint8
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
        tasks = {row.get("task_name", "beat_block_hammer") for row in records}
        assert len(tasks) == 1, "Evaluation validation requires one recorded task"
        task = tasks.pop()
        expected_pairs = {
            (row["seed"], row["instruction"]) for row in manifest[task]["episodes"]
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
    name = "validation.json" if require_complete else "partial_validation.json"
    (root / name).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--episodes", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.episodes)))
