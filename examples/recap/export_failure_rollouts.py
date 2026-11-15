# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Export failed SFT40k rollouts as complete three-camera H.264 videos."""

import io
import json
import os
import subprocess
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from rlinf.scheduler import Cluster, ComponentPlacement, Worker

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def encode_video(path, frames, fps=15):
    """Encode every recorded RGB frame and verify decoded count and duration."""
    assert frames and not path.exists(), path
    h, w = frames[0][0].shape[:2]
    width, height = w * 3, h + 32
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for index, images in enumerate(frames):
            panel = Image.new("RGB", (width, height), "black")
            draw = ImageDraw.Draw(panel)
            for j, (name, image) in enumerate(zip(CAMERAS, images, strict=True)):
                panel.paste(Image.fromarray(image), (j * w, 32))
                draw.text((j * w + 5, 5), name, fill="white")
            draw.text(
                (width - 180, 17), f"frame {index}/{len(frames) - 1}", fill="white"
            )
            proc.stdin.write(panel.tobytes())
    finally:
        proc.stdin.close()
    assert proc.wait() == 0, path
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,width,height:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    assert int(probe["streams"][0]["nb_read_frames"]) == len(frames), probe
    assert abs(float(probe["format"]["duration"]) - len(frames) / fps) < 0.15, probe
    return {
        "video": path.name,
        "frames": len(frames),
        "fps": fps,
        "duration_seconds": float(probe["format"]["duration"]),
        "bytes": path.stat().st_size,
    }


def export_recorded_hammer(root):
    """Use the first ten sorted failures in the existing independent val set."""
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    if (root / "DONE.json").exists():
        return
    source = Path("/data2/sjy/recap/beat_block_hammer/rollout4096_v2/datasets/rollout")
    chosen = []
    for log in sorted(source.glob("rank_*/episodes.jsonl")):
        for line in log.read_text().splitlines():
            e = json.loads(line)
            if e["split"] == "val" and not e["success"]:
                chosen.append((log.parent / "val", e))
    assert len(chosen) >= 10
    records = []
    for index, (directory, e) in enumerate(chosen[:10]):
        info = json.loads((directory / "meta/info.json").read_text())
        ep = e["episode_index"]
        parquet = directory / info["data_path"].format(
            episode_chunk=ep // info["chunks_size"], episode_index=ep
        )
        table = pq.read_table(
            parquet, columns=["observation.images." + c for c in CAMERAS]
        )
        frames = []
        for i in range(table.num_rows):
            frames.append(
                [
                    np.asarray(
                        Image.open(
                            io.BytesIO(
                                table["observation.images." + c][i].as_py()["bytes"]
                            )
                        ).convert("RGB")
                    )
                    for c in CAMERAS
                ]
            )
        path = root / f"failure_{index + 1:02d}_seed{e['seed']}.mp4"
        record = encode_video(path, frames, info["fps"])
        record.update(
            e,
            task_name="beat_block_hammer",
            rerendered=False,
            source_parquet=str(parquet),
            includes_post_action_terminal_image=False,
        )
        (root / (path.stem + ".json")).write_text(json.dumps(record, indent=2))
        records.append(record)
    (root / "manifest.json").write_text(json.dumps(records, indent=2))
    (root / "DONE.json").write_text(json.dumps({"failure_videos": len(records)}))


def rgb_views(obs):
    images = [obs["main_images"][0], *obs["wrist_images"][0]]
    return [x.detach().cpu().numpy().copy() for x in images]


class FailureVideoCollector(Worker):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        from examples.recap.collect_robotwin import reset_collection_batch
        from rlinf.envs.robotwin.collection_utils import (
            collection_task_name,
            install_fresh_collection_reset,
        )
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
        from rlinf.models.embodiment.openpi_cfg import get_model
        from rlinf.models.embodiment.openpi_cfg.checkpoint import load_pt_policy

        cfg = self.cfg
        rank = self._rank
        torch.cuda.set_device(0)
        install_fresh_collection_reset()
        env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
        task = collection_task_name(env_cfg)
        root = Path(cfg.collect.output) / f"rank_{rank:02d}"
        root.mkdir(parents=True, exist_ok=False)
        # All parameters are loaded strictly below; skip discarded random initialization.
        from transformers.modeling_utils import no_init_weights

        assert cfg.actor.model.strict_checkpoint
        with no_init_weights():
            model = get_model(cfg.actor.model)
        if cfg.collect.policy_checkpoint:
            model.checkpoint_provenance = load_pt_policy(
                model, Path(cfg.collect.policy_checkpoint)
            )
        model = model.to("cuda:0").eval()
        (root / "policy.json").write_text(
            json.dumps(model.checkpoint_provenance, indent=2)
        )
        env = RoboTwinEnv(
            cfg=env_cfg,
            num_envs=1,
            seed_offset=rank,
            total_num_processes=self._world_size,
            worker_info=self.worker_info,
            record_metrics=True,
        )
        target = 2 if rank < 2 else 1
        failures = 0
        try:
            for attempt in range(cfg.collect.episodes_per_worker):
                obs, _ = reset_collection_batch(env, attempt)
                seed = int(env.venv.envs[0].task.ep_num)
                instruction = str(obs["task_descriptions"][0])
                assert instruction.strip()
                noise_seed = cfg.collect.seed + rank * 10000 + attempt
                torch.manual_seed(noise_seed)
                np.random.seed(noise_seed)
                frames = []
                success = truncated = False
                for start in range(
                    0, env_cfg.max_episode_steps, cfg.collect.action_chunk
                ):
                    with torch.inference_mode():
                        actions, _ = model.predict_action_batch(obs, mode="eval")
                    for j in range(
                        min(cfg.collect.action_chunk, env_cfg.max_episode_steps - start)
                    ):
                        frames.append(rgb_views(obs))
                        obs, _, term, trunc, _ = env.step(
                            actions[:, j], auto_reset=False
                        )
                        success, truncated = bool(term[0]), bool(trunc[0])
                        if success or truncated:
                            break
                    if success or truncated:
                        break
                summary = {
                    "task_name": task,
                    "seed": seed,
                    "requested_seed": int(env.reset_state_ids[0]),
                    "instruction": instruction,
                    "success": success,
                    "truncated": truncated,
                    "actions": len(frames),
                    "attempt": attempt,
                    "policy_noise_seed": noise_seed,
                    "policy": model.checkpoint_provenance["path"],
                    "rerendered": True,
                }
                if not success:
                    frames.append(rgb_views(obs))
                    path = root / f"failure_{failures + 1:02d}_seed{seed}.mp4"
                    summary.update(encode_video(path, frames, cfg.collect.fps))
                    summary["includes_post_action_terminal_image"] = True
                    (root / (path.stem + ".json")).write_text(
                        json.dumps(summary, indent=2)
                    )
                    failures += 1
                with (root / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(summary) + "\n")
                self.log_info(str(summary))
                (root / "progress.json").write_text(
                    json.dumps(
                        {
                            "attempts": attempt + 1,
                            "failures": failures,
                            "target": target,
                        }
                    )
                )
                if failures == target:
                    break
            assert failures == target, (task, rank, failures, target)
            (root / "DONE.json").write_text(json.dumps({"failures": failures}))
        finally:
            # Keep native renderer/model alive until this isolated Ray session exits.
            # Explicit per-worker teardown can race Vulkan background threads.
            self._retained_env = env
            self._retained_model = model


@hydra.main(version_base="1.1", config_path="config", config_name="robotwin_collect")
def main(cfg):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster)
    workers = FailureVideoCollector.create_group(cfg).launch(
        cluster,
        name="FailureVideoCollector",
        placement_strategy=placement.get_strategy("collector"),
    )
    workers.run().wait()


if __name__ == "__main__":
    main()
