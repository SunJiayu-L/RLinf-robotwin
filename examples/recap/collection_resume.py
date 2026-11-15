# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Resume committed collection episodes without loading historical RGB tables."""

import json
import math
import shutil
import time
from pathlib import Path


def read_summaries(root: Path, quota: int, validation_quota: int) -> list[dict]:
    """Require a contiguous committed prefix with independent split indices."""
    path = root / "episodes.jsonl"
    rows = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    if len(rows) > quota:
        raise ValueError(f"Too many committed episodes in {root}")
    counts = {"train": 0, "val": 0}
    for position, row in enumerate(rows):
        split = "val" if position < validation_quota else "train"
        if row["split"] != split or row["episode_index"] != counts[split]:
            raise ValueError(
                f"Non-contiguous collection checkpoint: {root}, {position}"
            )
        counts[split] += 1
    return rows


def prepare_resume(root: Path, quota: int, validation_quota: int) -> dict:
    """Quarantine interrupted writes beyond the episode journal's commit point.

    Completed episodes are never removed. Metadata can only be rolled back from
    an uncommitted suffix, whose original files are retained in the quarantine.
    """
    rows = read_summaries(root, quota, validation_quota)
    quarantine = root / "recovery" / time.strftime("%Y%m%d_%H%M%S")
    moved = []

    def preserve(path: Path, *, move: bool) -> None:
        target = quarantine / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            shutil.move(str(path), target)
        else:
            shutil.copy2(path, target)
        moved.append(str(path.relative_to(root)))

    for split in ["val", "train"]:
        selected = [row for row in rows if row["split"] == split]
        directory = root / split
        if not directory.exists():
            if selected:
                raise FileNotFoundError(f"Missing committed dataset: {directory}")
            continue
        if not selected:
            preserve(directory, move=True)
            continue
        info_path = directory / "meta/info.json"
        info = json.loads(info_path.read_text())
        if info["total_episodes"] < len(selected):
            raise ValueError(f"Metadata is behind committed journal: {directory}")
        for name in ["episodes.jsonl", "episodes_stats.jsonl"]:
            path = directory / "meta" / name
            metadata = (
                [json.loads(line) for line in path.read_text().splitlines()]
                if path.exists()
                else []
            )
            prefix = [row for row in metadata if row["episode_index"] < len(selected)]
            if [row["episode_index"] for row in prefix] != list(range(len(selected))):
                raise ValueError(f"Missing committed metadata: {path}")
            if len(prefix) != len(metadata):
                preserve(path, move=False)
                path.write_text("".join(json.dumps(row) + "\n" for row in prefix))
        expected_frames = sum(row["frames"] for row in selected)
        if (
            info["total_episodes"] != len(selected)
            or info["total_frames"] != expected_frames
        ):
            preserve(info_path, move=False)
            info.update(
                total_episodes=len(selected),
                total_frames=expected_frames,
                total_chunks=math.ceil(len(selected) / info["chunks_size"]),
                splits={"train": f"0:{len(selected)}"},
            )
            info_path.write_text(json.dumps(info, indent=2) + "\n")
        for row in selected:
            relative = info["data_path"].format(
                episode_index=row["episode_index"],
                episode_chunk=row["episode_index"] // info["chunks_size"],
            )
            if (
                not (directory / relative).is_file()
                or not (root / row["terminal_observation"]).is_file()
            ):
                raise FileNotFoundError(f"Missing committed trajectory: {root}, {row}")
        for path in directory.glob("data/**/*.parquet"):
            if int(path.stem.split("_")[-1]) >= len(selected):
                preserve(path, move=True)
        for path in (root / "terminal_observations").glob(f"{split}_*.npz"):
            if int(path.stem.split("_")[-1]) >= len(selected):
                preserve(path, move=True)
        if (directory / "images").exists():
            preserve(directory / "images", move=True)
    return {"episodes": len(rows), "quarantined": moved}


def resume_writer(root: Path, repo_id: str):
    """Open LeRobot's recording interface using metadata and an empty Arrow cache."""
    from lerobot.common.datasets.lerobot_dataset import (
        LeRobotDataset,
        LeRobotDatasetMetadata,
    )

    writer = LeRobotDataset.__new__(LeRobotDataset)
    writer.meta = LeRobotDatasetMetadata(repo_id, root)
    writer.repo_id = repo_id
    writer.root = root
    writer.revision = None
    writer.tolerance_s = 1e-4
    writer.image_writer = None
    writer.episode_buffer = writer.create_episode_buffer()
    writer.episodes = None
    writer.hf_dataset = writer.create_hf_dataset()
    writer.image_transforms = None
    writer.delta_timestamps = None
    writer.delta_indices = None
    writer.episode_data_index = None
    writer.video_backend = "pyav"
    writer.start_image_writer(num_threads=2)
    return writer


def seek_batch(env, batch: int) -> None:
    """Advance seed/manifest cursors without replaying completed simulation."""
    for _ in range(batch):
        env.update_reset_state_ids()
