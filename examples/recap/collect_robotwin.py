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

"""Collect full-resolution-time RoboTwin experience with RLinf GPU workers.

Each worker produces an independent LeRobot shard. The offline RLinf mixture
reads these shards directly; task/seed/policy provenance is saved per episode.
"""

import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.scheduler import Cluster, ComponentPlacement, Worker


def reset_collection_batch(env, batch):
    """The environment constructor already selects the first manifest episode."""
    if batch:
        env.update_reset_state_ids()
    return env.reset()


class RoboTwinCollector(Worker):
    """One policy replica and vector simulator per allocated RLinf GPU."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def run(self):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        from examples.recap.collection_resume import (
            read_summaries,
            resume_writer,
            seek_batch,
        )
        from rlinf.envs.robotwin.collection_utils import (
            append_chunk_frames,
            collection_task_name,
            install_fresh_collection_reset,
            release_saved_arrow_cache,
        )
        from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
        from rlinf.models.embodiment.openpi_cfg import get_model
        from rlinf.models.embodiment.openpi_cfg.checkpoint import load_pt_policy

        install_fresh_collection_reset()
        cfg = self.cfg
        rank = self._rank
        root = Path(cfg.collect.output) / f"rank_{rank:02d}"
        resuming = bool(cfg.collect.get("resume", False))
        root.mkdir(parents=True, exist_ok=resuming)
        quota = cfg.collect.episodes_per_worker
        nenv = cfg.collect.num_envs_per_worker
        assert quota % nenv == 0
        summaries = read_summaries(
            root, quota, cfg.collect.validation_episodes_per_worker
        )
        if len(summaries) == quota:
            (root / "DONE.json").write_text(
                json.dumps(
                    {
                        "episodes": quota,
                        "successes": sum(r["success"] for r in summaries),
                    }
                )
            )
            return
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        torch.manual_seed(cfg.collect.seed + rank)
        np.random.seed(cfg.collect.seed + rank)
        from transformers.modeling_utils import no_init_weights

        assert cfg.actor.model.strict_checkpoint
        with no_init_weights():
            model = get_model(cfg.actor.model)
        if cfg.collect.get("policy_checkpoint"):
            model.checkpoint_provenance = load_pt_policy(
                model, Path(cfg.collect.policy_checkpoint)
            )
        model = model.to(device).eval()
        if (root / "policy.json").exists():
            previous = json.loads((root / "policy.json").read_text())
            if previous != model.checkpoint_provenance:
                raise ValueError(f"Cannot resume with different policy: {root}")
        (root / "policy.json").write_text(
            json.dumps(model.checkpoint_provenance, indent=2)
        )
        env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.env.train, resolve=True))
        task_name = collection_task_name(env_cfg)
        env = RoboTwinEnv(
            cfg=env_cfg,
            num_envs=cfg.collect.num_envs_per_worker,
            seed_offset=rank,
            total_num_processes=self._world_size,
            worker_info=self.worker_info,
            record_metrics=True,
        )
        datasets = {}
        try:
            start_batch = len(summaries) // nenv
            seek_batch(env, start_batch)
            batch_limit = int(cfg.collect.get("max_batches_per_run", quota // nenv))
            if batch_limit <= 0:
                raise ValueError("max_batches_per_run must be positive")
            end_batch = min(quota // nenv, start_batch + batch_limit)
            self.log_info(
                f"Resuming {len(summaries)}/{quota} episodes; batches {start_batch}:{end_batch}"
            )
            for batch in range(start_batch, end_batch):
                noise_seed = cfg.collect.seed + rank * 10000 + batch
                obs, _ = reset_collection_batch(env, batch - start_batch)
                torch.manual_seed(noise_seed)
                np.random.seed(noise_seed)
                requested_seeds = env.reset_state_ids.tolist()
                seeds = [int(sub.task.ep_num) for sub in env.venv.envs]
                for requested, actual in zip(requested_seeds, seeds, strict=True):
                    if not requested <= actual < requested + 1024:
                        raise ValueError(
                            f"Unexpected scene seed fallback: {requested} -> {actual}"
                        )
                frames = [[] for _ in range(nenv)]
                active = np.ones(nenv, dtype=bool)
                success = np.zeros(nenv, dtype=bool)
                truncated = np.zeros(nenv, dtype=bool)
                executed_chunk_lengths = [[] for _ in range(nenv)]
                terminal_observations = [None for _ in range(nenv)]
                for start in range(
                    0, env_cfg.max_episode_steps, cfg.collect.action_chunk
                ):
                    with torch.inference_mode():
                        actions, _ = model.predict_action_batch(obs, mode="eval")
                    actions = actions[
                        :,
                        : min(
                            cfg.collect.action_chunk, env_cfg.max_episode_steps - start
                        ),
                    ]
                    post_obs, _, terms, truncs, infos = env.chunk_step(
                        actions, record_full_obs=True
                    )
                    counts = infos[-1]["executed_action_count"]
                    append_chunk_frames(frames, obs, actions, post_obs, counts, active)
                    term_array = terms.any(dim=1).cpu().numpy().astype(bool)
                    trunc_array = truncs.any(dim=1).cpu().numpy().astype(bool)
                    for index in np.flatnonzero(active):
                        executed_chunk_lengths[index].append(int(counts[index]))
                        if term_array[index] or trunc_array[index]:
                            terminal_observations[index] = {
                                "state": post_obs[-1]["states"][index]
                                .cpu()
                                .numpy()
                                .copy(),
                                "images": np.stack(
                                    [
                                        post_obs[-1]["main_images"][index]
                                        .cpu()
                                        .numpy(),
                                        *post_obs[-1]["wrist_images"][index]
                                        .cpu()
                                        .numpy(),
                                    ]
                                ),
                            }
                    success |= active & term_array
                    truncated |= active & trunc_array
                    active &= ~(term_array | trunc_array)
                    obs = post_obs[-1]
                    if not active.any():
                        break
                for index in range(nenv):
                    episode_number = batch * nenv + index
                    split = (
                        "val"
                        if episode_number < cfg.collect.validation_episodes_per_worker
                        else "train"
                    )
                    if episode_number < len(summaries):
                        previous = summaries[episode_number]
                        replay = {
                            "requested_seed": requested_seeds[index],
                            "seed": seeds[index],
                            "instruction": frames[index][0]["task"],
                            "policy_noise_seed": noise_seed,
                            "success": bool(success[index]),
                            "frames": len(frames[index]),
                            "executed_chunk_lengths": executed_chunk_lengths[index],
                        }
                        if any(
                            previous.get(key) != value for key, value in replay.items()
                        ):
                            raise ValueError(
                                f"Partial-batch replay differs: {root}, {replay}"
                            )
                        self.log_info(
                            f"Verified already committed partial-batch episode {episode_number}"
                        )
                        continue
                    if split not in datasets:
                        if (root / split).exists():
                            datasets[split] = resume_writer(
                                root / split, f"robotwin_recap_{rank}_{split}"
                            )
                    if split not in datasets:
                        image = frames[index][0]["observation.images.cam_high"]
                        features = {
                            "observation.state": {
                                "dtype": "float32",
                                "shape": (14,),
                                "names": None,
                            },
                            "action": {
                                "dtype": "float32",
                                "shape": (14,),
                                "names": None,
                            },
                            "is_success": {
                                "dtype": "bool",
                                "shape": (1,),
                                "names": None,
                            },
                            **{
                                f"observation.images.{cam}": {
                                    "dtype": "image",
                                    "shape": image.shape,
                                    "names": ["height", "width", "channels"],
                                }
                                for cam in [
                                    "cam_high",
                                    "cam_left_wrist",
                                    "cam_right_wrist",
                                ]
                            },
                        }
                        datasets[split] = LeRobotDataset.create(
                            repo_id=f"robotwin_recap_{rank}_{split}",
                            root=root / split,
                            fps=cfg.collect.fps,
                            robot_type="aloha",
                            features=features,
                            use_videos=False,
                            image_writer_threads=2,
                        )
                    dataset = datasets[split]
                    episode_id = dataset.meta.total_episodes
                    for frame in frames[index]:
                        frame["is_success"] = np.array([success[index]], dtype=bool)
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    release_saved_arrow_cache(dataset)
                    if terminal_observations[index] is None:
                        raise RuntimeError(
                            "Collection ended without a terminal observation"
                        )
                    terminal_path = (
                        root / "terminal_observations" / f"{split}_{episode_id:06d}.npz"
                    )
                    terminal_path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(terminal_path, **terminal_observations[index])
                    summary = {
                        "collection_episode_index": episode_number,
                        "episode_index": episode_id,
                        "split": split,
                        "seed": seeds[index],
                        "requested_seed": requested_seeds[index],
                        "scene_info": env.venv.envs[index].task.get_info(),
                        "task_name": task_name,
                        "success": bool(success[index]),
                        "truncated": bool(truncated[index]),
                        "frames": len(frames[index]),
                        "instruction": frames[index][0].get(
                            "task", env._observed_instructions[index]
                        ),
                        "policy": model.checkpoint_provenance["path"],
                        "policy_noise_seed": noise_seed,
                        "execution_mode": "native_full_chunk_topp",
                        "observation_alignment": "pre_action; terminal observation stored separately",
                        "executed_chunk_lengths": executed_chunk_lengths[index],
                        "terminal_observation": str(terminal_path.relative_to(root)),
                    }
                    summaries.append(summary)
                    with (root / "episodes.jsonl").open("a") as stream:
                        stream.write(json.dumps(summary) + "\n")
                    self.log_info(f"Collected {summary}")
                    import resource

                    progress = {
                        "episodes": len(summaries),
                        "quota": quota,
                        "train": sum(row["split"] == "train" for row in summaries),
                        "val": sum(row["split"] == "val" for row in summaries),
                        "max_rss_mib": resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        / 1024,
                        "saved_arrow_bytes": dataset.hf_dataset._data.nbytes,
                        "last_seed": seeds[index],
                    }
                    (root / "progress.json").write_text(json.dumps(progress, indent=2))
                meminfo = {
                    line.split(":")[0]: int(line.split(":")[1].split()[0])
                    for line in Path("/proc/meminfo").read_text().splitlines()
                }
                if meminfo["MemAvailable"] < 100 * 1024 * 1024:
                    raise MemoryError(
                        "Collection stopped with complete episodes: system available RAM below 100 GiB"
                    )

            marker = "DONE.json" if len(summaries) == quota else "SEGMENT_DONE.json"
            (root / marker).write_text(
                json.dumps(
                    {
                        "episodes": len(summaries),
                        "successes": sum(r["success"] for r in summaries),
                    }
                )
            )
        finally:
            self._retained_env = env
            self._retained_model = model
            for dataset in datasets.values():
                if dataset.image_writer is not None:
                    dataset.image_writer.stop()


@hydra.main(version_base="1.1", config_path="config", config_name="robotwin_collect")
def main(cfg):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster)
    workers = RoboTwinCollector.create_group(cfg).launch(
        cluster,
        name="RECAPCollector",
        placement_strategy=placement.get_strategy("collector"),
    )
    workers.run().wait()


if __name__ == "__main__":
    main()
