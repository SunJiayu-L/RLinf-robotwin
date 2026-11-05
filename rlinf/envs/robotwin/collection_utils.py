# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Bounded LeRobot writing and fresh scene-conditioned collection instructions."""


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
