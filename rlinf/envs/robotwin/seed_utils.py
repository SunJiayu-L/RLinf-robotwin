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

from typing import Any

import torch


def partition_success_seeds(
    success_seeds: torch.Tensor,
    *,
    base_seed: int,
    seed_offset: int,
    total_num_processes: int,
    num_group: int,
) -> torch.Tensor:
    """Shuffle success seeds globally and return the non-overlapping worker slice."""
    global_generator = torch.Generator()
    global_generator.manual_seed(base_seed)
    shuffled_indices = torch.randperm(success_seeds.numel(), generator=global_generator)
    shuffled_seeds = success_seeds[shuffled_indices]

    seeds_per_worker = shuffled_seeds.numel() // total_num_processes
    start = seed_offset * seeds_per_worker
    end = start + seeds_per_worker
    worker_seeds = shuffled_seeds[start:end]

    keep_count = (worker_seeds.numel() // num_group) * num_group
    return worker_seeds[:keep_count]


def partition_episode_manifest(
    episodes: list[dict[str, Any]],
    *,
    seed_offset: int,
    total_num_processes: int,
    num_group: int,
) -> list[dict[str, Any]]:
    """Return one worker's ordered, non-overlapping episode manifest slice.

    Unlike training seed partitioning, paired evaluation must preserve the exact
    order from the reference evaluation and must never shuffle or silently drop
    different numbers of episodes on different workers.
    """
    if total_num_processes <= 0:
        raise ValueError("total_num_processes must be positive")
    if not 0 <= seed_offset < total_num_processes:
        raise ValueError(
            f"seed_offset must be in [0, {total_num_processes}), got {seed_offset}"
        )
    if num_group <= 0:
        raise ValueError("num_group must be positive")
    if len(episodes) % total_num_processes != 0:
        raise ValueError(
            "Episode manifest size must be divisible by total_num_processes: "
            f"{len(episodes)} % {total_num_processes} != 0"
        )

    episodes_per_worker = len(episodes) // total_num_processes
    if episodes_per_worker % num_group != 0:
        raise ValueError(
            "Each worker's episode count must be divisible by its number of "
            f"environment groups: {episodes_per_worker} % {num_group} != 0"
        )

    start = seed_offset * episodes_per_worker
    end = start + episodes_per_worker
    return episodes[start:end]
