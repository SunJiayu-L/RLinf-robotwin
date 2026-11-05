#!/usr/bin/env python3
"""Compute pi05 ALOHA normalization for the default eight RoboTwin tasks.

Reads only state/action parquet columns. The transformed state and the padded
50-step action windows match pi05_aloha_robotwin normalization inputs.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


TASKS = (
    "beat_block_hammer",
    "place_phone_stand",
    "pick_dual_bottles",
    "lift_pot",
    "move_can_pot",
    "place_a2b_left",
    "place_empty_cup",
    "handover_mic",
)
FLIP = np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])
DELTA = np.array([True] * 6 + [False] + [True] * 6 + [False])
HORIZON = 50
BINS = 5000


def task_ranges(conversion_log: Path, total_episodes: int) -> dict[str, tuple[int, int]]:
    text = conversion_log.read_text(errors="replace")[:20000]
    names = re.findall(r"^  - demo_clean/([^/]+)/aloha_agilex: (\d+) files", text, re.M)
    if len(names) != 23 or len({name for name, _ in names}) != 23:
        raise ValueError("Expected 23 unique tasks in converter's Matched targets list")
    ranges = {}
    start = 0
    for name, count_text in names:
        count = int(count_text)
        ranges[name] = (start, start + count)
        start += count
    if start != total_episodes or any(name not in ranges for name in TASKS):
        raise ValueError(f"Conversion mapping does not match dataset: {start=} {total_episodes=}")
    return ranges


def transform_episode(path: str) -> tuple[np.ndarray, np.ndarray]:
    columns = pq.read_table(path, columns=["observation.state", "action"]).to_pydict()
    raw_state = np.asarray(columns["observation.state"], dtype=np.float64)
    raw_action = np.asarray(columns["action"], dtype=np.float64)
    if raw_state.ndim != 2 or raw_state.shape[1] != 14 or raw_action.shape != raw_state.shape:
        raise ValueError(f"Bad state/action shape in {path}: {raw_state.shape}, {raw_action.shape}")
    if not len(raw_state) or not np.isfinite(raw_state).all() or not np.isfinite(raw_action).all():
        raise ValueError(f"Empty or nonfinite episode: {path}")

    # AlohaInputs(adapt_to_pi=True): flip joint signs and convert gripper units.
    state = raw_state * FLIP
    linear = raw_state[:, [6, 13]] * (0.05800 - 0.01844) + 0.01844
    cosine_arg = (0.022**2 + linear**2 - 0.036**2) / (2 * 0.022 * linear)
    state[:, [6, 13]] = (np.arcsin(np.clip(cosine_arg, -1, 1)) - 0.5476) / (1.6296 - 0.5476)

    # LeRobot delta timestamps use the current frame plus 0..49, clamped at
    # the episode's last frame. This includes padded actions in the statistics.
    indices = np.minimum(np.arange(len(raw_action))[:, None] + np.arange(HORIZON)[None, :], len(raw_action) - 1)
    actions = raw_action[indices] * FLIP
    actions[..., [6, 13]] = raw_action[indices][..., [6, 13]] * (1.4910 + 0.6213) - 0.6213 - 0.5476
    actions[..., DELTA] -= state[:, None, DELTA]
    return state, actions.reshape(-1, 14)


def first_pass(paths: list[str]) -> dict:
    result = {key: {"count": 0, "sum": np.zeros(14), "sumsq": np.zeros(14),
                    "min": np.full(14, np.inf), "max": np.full(14, -np.inf)}
              for key in ("state", "actions")}
    for path in paths:
        state, actions = transform_episode(path)
        for key, values in (("state", state), ("actions", actions)):
            part = result[key]
            part["count"] += len(values)
            part["sum"] += values.sum(axis=0)
            part["sumsq"] += np.square(values).sum(axis=0)
            part["min"] = np.minimum(part["min"], values.min(axis=0))
            part["max"] = np.maximum(part["max"], values.max(axis=0))
    return result


def histogram_pass(paths: list[str], edges: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {key: np.zeros((14, BINS), dtype=np.int64) for key in ("state", "actions")}
    for path in paths:
        state, actions = transform_episode(path)
        for key, values in (("state", state), ("actions", actions)):
            for dim in range(14):
                result[key][dim] += np.histogram(values[:, dim], bins=edges[key][dim])[0]
    return result


def run_pool(fn, shards: list[list[str]], workers: int, *args):
    with futures.ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(fn, shard, *args) for shard in shards]
        for completed, job in enumerate(futures.as_completed(pending), 1):
            yield job.result()
            if completed % 4 == 0 or completed == len(pending):
                print(f"{fn.__name__}: {completed}/{len(pending)} shards", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--conversion-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--sample-per-task", type=int, default=500,
                        help="Episodes per task; use 1 for a quick pipeline check")
    args = parser.parse_args()
    if not 1 <= args.workers <= (os.cpu_count() or 1):
        raise ValueError("workers must be between 1 and available logical CPUs")

    info = json.loads((args.dataset / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or info["total_episodes"] != 11500:
        raise ValueError("Unexpected source LeRobot dataset")
    ranges = task_ranges(args.conversion_log, info["total_episodes"])
    if not 1 <= args.sample_per_task <= 500:
        raise ValueError("sample-per-task must be in 1..500")
    selected = [(name, index) for name in TASKS
                for index in range(ranges[name][0], ranges[name][0] + args.sample_per_task)]
    paths = [str(args.dataset / info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index))
             for _, index in selected]
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} selected parquets; first: {missing[0]}")
    shards = [paths[i::args.workers] for i in range(args.workers)]
    print(f"Selected {len(paths)} episodes from {len(TASKS)} tasks; starting {args.workers} CPU workers", flush=True)

    totals = {key: {"count": 0, "sum": np.zeros(14), "sumsq": np.zeros(14),
                    "min": np.full(14, np.inf), "max": np.full(14, -np.inf)}
              for key in ("state", "actions")}
    for partial in run_pool(first_pass, shards, args.workers):
        for key in totals:
            dst, src = totals[key], partial[key]
            dst["count"] += src["count"]
            dst["sum"] += src["sum"]
            dst["sumsq"] += src["sumsq"]
            dst["min"] = np.minimum(dst["min"], src["min"])
            dst["max"] = np.maximum(dst["max"], src["max"])

    edges = {}
    for key, values in totals.items():
        span = values["max"] - values["min"]
        pad = np.where(span > 0, 1e-10, 1e-8)
        edges[key] = np.array([np.linspace(lo - p, hi + p, BINS + 1)
                               for lo, hi, p in zip(values["min"], values["max"], pad, strict=True)])
    histograms = {key: np.zeros((14, BINS), dtype=np.int64) for key in totals}
    for partial in run_pool(histogram_pass, shards, args.workers, edges):
        for key in histograms:
            histograms[key] += partial[key]

    norm_stats = {}
    for key, values in totals.items():
        count = values["count"]
        if count < 2 or not np.all(histograms[key].sum(axis=1) == count):
            raise ValueError(f"Statistics count mismatch for {key}: {count}")
        mean = values["sum"] / count
        std = np.sqrt(np.maximum(0, values["sumsq"] / count - mean**2))
        cumulative = np.cumsum(histograms[key], axis=1)
        quantiles = {}
        for label, quantile in (("q01", 0.01), ("q99", 0.99)):
            positions = [np.searchsorted(cumulative[dim], quantile * count) for dim in range(14)]
            quantiles[label] = [float(edges[key][dim, min(pos, BINS)]) for dim, pos in enumerate(positions)]
        norm_stats[key] = {"mean": mean.tolist(), "std": std.tolist(), **quantiles}
        print(f"{key}: {count:,} samples; mean/std/q01/q99 all 14D", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_dataset": str(args.dataset.resolve()),
        "config_name": "pi05_aloha_robotwin", "action_horizon": HORIZON,
        "workers": args.workers, "tasks": list(TASKS),
        "sample_per_task": args.sample_per_task,
        "episode_ranges": {name: list(ranges[name]) for name in TASKS},
        "total_episodes": len(paths),
        "state_samples": totals["state"]["count"],
        "action_samples": totals["actions"]["count"],
        "quantile_bins": BINS,
    }
    for filename, value in (("norm_stats.json", {"norm_stats": norm_stats}), ("manifest.json", manifest)):
        target = args.output / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(target)
    print(f"Saved {args.output / 'norm_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
