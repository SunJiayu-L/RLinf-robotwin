# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded LeRobot writing and fresh scene-conditioned collection instructions."""

import numpy as np


def append_chunk_frames(
    frames: list[list[dict]],
    initial_obs: dict,
    actions: np.ndarray,
    post_observations: list[dict],
    executed_counts: list[int],
    active: np.ndarray,
) -> None:
    """Pair pre-action observations with executed actions, excluding terminal padding.

    Native full-observation execution returns post-action observations padded to
    the requested chunk length. The terminal observation is retained separately
    by the collector, rather than becoming an extra fictitious action frame.
    """
    chunk_length = actions.shape[1]
    if len(post_observations) != chunk_length:
        raise ValueError("Full-observation count must equal the requested chunk length")
    if len(executed_counts) != len(frames) or len(active) != len(frames):
        raise ValueError(
            "Executed counts and active flags must match environment count"
        )
    for index in np.flatnonzero(active):
        count = int(executed_counts[index])
        if not 0 < count <= chunk_length:
            raise ValueError(
                f"Invalid executed action count for active env {index}: {count}"
            )
        for action_index in range(count):
            obs = (
                initial_obs
                if action_index == 0
                else post_observations[action_index - 1]
            )
            images = [obs["main_images"][index], *obs["wrist_images"][index]]
            frames[index].append(
                {
                    "observation.state": obs["states"][index]
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=True),
                    "action": np.asarray(
                        actions[index, action_index], dtype=np.float32
                    ).copy(),
                    **{
                        f"observation.images.{name}": image.cpu().numpy().copy()
                        for name, image in zip(
                            ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                            images,
                            strict=True,
                        )
                    },
                    "task": obs["task_descriptions"][index],
                }
            )


def collection_task_name(env_cfg) -> str:
    """Require one consistent task per collection shard."""
    task = str(env_cfg["task_config"]["task_name"])
    tasks = [str(name) for name in env_cfg.get("task_names", [task])]
    if tasks != [task]:
        raise ValueError(
            f"Collection requires a single consistent task: {tasks}, {task}"
        )
    return task


def release_saved_arrow_cache(writer) -> None:
    """Drop saved in-memory image tables; episode metadata and Parquet stay intact."""
    from datasets import Dataset

    writer.hf_dataset = Dataset.from_dict(
        {key: [] for key in writer.hf_features}, features=writer.hf_features
    )


def make_fresh_reset(native_reset):
    """Refresh descriptions after reset, when the new scene information is known."""

    def reset(self, env_seed=None, instruction=None):
        result = native_reset(self, env_seed=env_seed, instruction=instruction)
        with self.global_lock:
            with self.lock:
                self.env_seed = int(self.task.ep_num)
                self.episode_info_list = [self.task.get_info()]
                text = (
                    instruction
                    if instruction is not None
                    else self.create_instruction()
                )
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("Reset produced an empty RoboTwin instruction")
                self.instruction = text
                self.args["instruction"] = text
                self.task.set_instruction(text)
        return result

    return reset


def install_fresh_collection_reset() -> None:
    """Install only in collection workers; preserve native simulator reset behavior."""
    from robotwin.envs.vector_env import SubEnv

    if not getattr(SubEnv, "_recap_fresh_reset", False):
        SubEnv.reset = make_fresh_reset(SubEnv.reset)
        SubEnv._recap_fresh_reset = True
