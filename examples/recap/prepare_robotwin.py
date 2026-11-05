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

"""Create immutable, episode-disjoint LeRobot views; do not rewrite SFT data."""

import argparse
import json
import shutil
from pathlib import Path


def create_view(source: Path, target: Path, episode_ids: list[int], task_name: str):
    """Link source media/parquet and copy only selected episode metadata."""
    if target.exists():
        raise FileExistsError(target)
    info = json.loads((source / "meta/info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (source / "meta/episodes.jsonl").read_text().splitlines()
    ]
    episodes = [row for row in episodes if row["episode_index"] in set(episode_ids)]
    if {r["episode_index"] for r in episodes} != set(episode_ids):
        raise ValueError("Missing source episodes")
    (target / "meta").mkdir(parents=True)
    for row in episodes:
        relative = info["data_path"].format(
            episode_chunk=row["episode_index"] // info["chunks_size"],
            episode_index=row["episode_index"],
        )
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to((source / relative).resolve())
    if (source / "videos").exists():
        (target / "videos").symlink_to(
            (source / "videos").resolve(), target_is_directory=True
        )
    for filename in ["tasks.jsonl", "stats.json"]:
        if (source / "meta" / filename).exists():
            shutil.copy2(source / "meta" / filename, target / "meta" / filename)
    epstats = source / "meta/episodes_stats.jsonl"
    if epstats.exists():
        selected = [json.loads(line) for line in epstats.read_text().splitlines()]
        selected = [r for r in selected if r["episode_index"] in set(episode_ids)]
        (target / "meta/episodes_stats.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in selected)
        )
    info.update(
        total_episodes=len(episodes),
        total_frames=sum(row["length"] for row in episodes),
        total_videos=len(episodes)
        * len([v for v in info["features"].values() if v["dtype"] == "video"]),
        splits={"train": f"0:{len(episodes)}"},
        recap_task_name=task_name,
    )
    (target / "meta/info.json").write_text(json.dumps(info, indent=2))
    (target / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in episodes)
    )
    (target / "selection.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "episodes": episode_ids,
                "task_name": task_name,
                "original_episode_ids_preserved": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/data2/sjy/datasets/lerobot/robotwin_demo_clean_aloha_agilex_23task_v21",
    )
    parser.add_argument(
        "--root", default="/data2/sjy/recap/beat_block_hammer/iter000/datasets"
    )
    args = parser.parse_args()
    # The validated 23-task conversion stores beat_block_hammer in episodes 0:500.
    source, root = Path(args.source), Path(args.root)
    create_view(source, root / "sft_train", list(range(450)), "beat_block_hammer")
    create_view(source, root / "sft_val", list(range(450, 500)), "beat_block_hammer")
    create_view(source, root / "sft_smoke", list(range(8)), "beat_block_hammer")
    print(json.dumps({"train": 450, "validation": 50, "smoke": 8, "root": str(root)}))
