#!/usr/bin/env python3
"""Filter a LeRobot dataset to keep specific episode index ranges.

Example usage:
    python toolkits/lerobot/filter_lerobot_dataset.py \
        --source-dir /path/to/robotwin_47_pi05 \
        --output-dir /path/to/robotwin_15_pi05 \
        --episodes "0-99,100-199,400-499"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_episode_ranges(spec: str) -> set[int]:
    """Parse comma-separated ranges like '0-99,100-199,400-499' into a set of ints."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        elif part:
            result.add(int(part))
    return result


def filter_lerobot_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    episode_indices: set[int],
) -> int:
    import copy
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = Path(source_dir)
    dst = Path(output_dir)

    # Load source metadata
    with open(src / "meta" / "info.json") as f:
        info: dict[str, Any] = json.load(f)
    chunks_size: int = info.get("chunks_size", 1000)

    episodes: list[dict] = []
    with open(src / "meta" / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))

    stats_by_old_idx: dict[int, dict] = {}
    stats_path = src / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        with open(stats_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    stats_by_old_idx[rec["episode_index"]] = rec["stats"]

    # Prepare output dirs
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(exist_ok=True)

    global_tasks: dict[str, int] = {}
    new_episodes: list[dict] = []
    new_stats: list[dict] = []
    global_frame_idx = 0
    new_ep_idx = 0

    for ep in episodes:
        old_idx: int = ep["episode_index"]
        if old_idx not in episode_indices:
            continue

        chunk = old_idx // chunks_size
        src_parquet = src / "data" / f"chunk-{chunk:03d}" / f"episode_{old_idx:06d}.parquet"
        if not src_parquet.is_file():
            print(f"[filter] WARNING: missing {src_parquet}, skipping", file=sys.stderr)
            continue

        table = pq.read_table(src_parquet)
        df = table.to_pandas()
        n = len(df)

        df["episode_index"] = new_ep_idx
        df["index"] = range(global_frame_idx, global_frame_idx + n)

        task = ep.get("tasks", ["unknown"])[0]
        if task not in global_tasks:
            global_tasks[task] = len(global_tasks)
        df["task_index"] = global_tasks[task]

        out_chunk = new_ep_idx // 1000
        out_dir = dst / "data" / f"chunk-{out_chunk:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode_{new_ep_idx:06d}.parquet"

        new_table = pa.Table.from_pandas(df, preserve_index=False)
        if table.schema.metadata:
            new_schema = new_table.schema.with_metadata(table.schema.metadata)
            new_table = new_table.cast(new_schema)
        pq.write_table(new_table, out_path)

        new_episodes.append({
            "episode_index": new_ep_idx,
            "tasks": ep.get("tasks", []),
            "length": n,
        })

        if old_idx in stats_by_old_idx:
            old_stats = copy.deepcopy(stats_by_old_idx[old_idx])
            # Update episode_index stats
            if "episode_index" in old_stats:
                count = old_stats["episode_index"].get("count", [1])
                old_stats["episode_index"] = {
                    "min": [new_ep_idx], "max": [new_ep_idx],
                    "mean": [float(new_ep_idx)], "std": [0.0], "count": count,
                }
            # Update global frame index stats
            if "index" in old_stats:
                old_min = old_stats["index"].get("min", [global_frame_idx])
                old_max = old_stats["index"].get("max", [global_frame_idx + n - 1])
                old_mean = old_stats["index"].get("mean", [float(global_frame_idx)])
                old_min_val = old_min[0] if isinstance(old_min, list) else old_min
                offset = global_frame_idx - old_min_val
                old_stats["index"] = {
                    "min": [old_min_val + offset],
                    "max": [(old_max[0] if isinstance(old_max, list) else old_max) + offset],
                    "mean": [(old_mean[0] if isinstance(old_mean, list) else old_mean) + offset],
                    "std": old_stats["index"].get("std", [0.0]),
                    "count": old_stats["index"].get("count", [1]),
                }
            new_stats.append({"episode_index": new_ep_idx, "stats": old_stats})

        global_frame_idx += n
        new_ep_idx += 1

        if new_ep_idx % 100 == 0 or new_ep_idx == len(episode_indices):
            print(f"[filter] {new_ep_idx}/{len(episode_indices)} episodes …")

    # Write metadata
    total_chunks = max(1, (new_ep_idx + 999) // 1000)
    info_out = dict(info)
    info_out.update({
        "total_episodes": new_ep_idx,
        "total_frames": global_frame_idx,
        "total_tasks": len(global_tasks),
        "total_chunks": total_chunks,
        "chunks_size": 1000,
        "splits": {"train": f"0:{new_ep_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    })
    info_out.pop("video_path", None)

    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(info_out, f, indent=4)

    with open(dst / "meta" / "episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    sorted_tasks = sorted(global_tasks.items(), key=lambda x: x[1])
    with open(dst / "meta" / "tasks.jsonl", "w") as f:
        for task, idx in sorted_tasks:
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    if new_stats:
        with open(dst / "meta" / "episodes_stats.jsonl", "w") as f:
            for s in new_stats:
                f.write(json.dumps(s) + "\n")
        print(f"[filter] Written episodes_stats.jsonl ({len(new_stats)} records)")

    print(f"[filter] Done: {new_ep_idx} episodes, {global_frame_idx} frames → {dst}")
    return new_ep_idx


def main() -> None:
    p = argparse.ArgumentParser(description="Filter a LeRobot dataset by episode ranges.")
    p.add_argument("--source-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--episodes",
        required=True,
        help="Comma-separated ranges, e.g. '0-99,100-199,400-499'",
    )
    args = p.parse_args()

    episode_indices = parse_episode_ranges(args.episodes)
    print(f"[filter] Keeping {len(episode_indices)} episodes from {args.source_dir}")
    n = filter_lerobot_dataset(args.source_dir, args.output_dir, episode_indices)
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
