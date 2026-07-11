import json
from pathlib import Path

import pyarrow.parquet as pq


DATASET = Path(
    "/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/"
    "robotwin_blocks_ranking_rgb_500_pi05_h16"
)
OUTPUT = Path(
    "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/my_docs/assets/"
    "blocks_rgb_training_frames"
)
EPISODES = (0, 1, 2, 3)


def image_bytes(value):
    if isinstance(value, dict):
        return value.get("bytes")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    return None


OUTPUT.mkdir(parents=True, exist_ok=True)
summary = []
for episode in EPISODES:
    path = DATASET / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(
        path,
        columns=["observation.images.cam_high", "observation.state", "action"],
    )
    images = table.column("observation.images.cam_high").combine_chunks()
    states = table.column("observation.state").combine_chunks()
    actions = table.column("action").combine_chunks()
    indices = list(range(max(0, len(images) - 5), len(images)))
    episode_dir = OUTPUT / f"episode_{episode:06d}"
    episode_dir.mkdir(exist_ok=True)
    written = []
    for index in indices:
        payload = image_bytes(images[index].as_py())
        if payload is None:
            raise RuntimeError(f"unsupported image value at episode={episode}, frame={index}")
        output = episode_dir / f"frame_{index:06d}.jpg"
        output.write_bytes(payload)
        written.append(str(output))
    summary.append(
        {
            "episode": episode,
            "length": len(images),
            "frames": written,
            "final_state": states[-1].as_py(),
            "final_action": actions[-1].as_py(),
            "final_grippers": {
                "state_left": states[-1].as_py()[6],
                "state_right": states[-1].as_py()[13],
                "action_left": actions[-1].as_py()[6],
                "action_right": actions[-1].as_py()[13],
            },
        }
    )

(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
