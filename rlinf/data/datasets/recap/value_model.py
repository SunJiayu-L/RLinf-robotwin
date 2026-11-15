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

"""Value-model datasets, transforms, and dataloader helpers for ReCap."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import openpi.models.model as _openpi_model
import openpi.transforms as _openpi_transforms
import pyarrow.parquet as pq
import torch
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from openpi.transforms import DataTransformFn
from torch.utils.data import Dataset

from rlinf.models.embodiment.openpi.policies import franka_policy, libero_policy

from .common import BaseDataLoaderImpl, ReCapMixtureDataset
from .robotwin import normalized_return, robotwin_value_input
from .utils import (
    decode_image_struct_batch,
    load_returns_sidecar,
    load_task_descriptions,
)

logger = logging.getLogger(__name__)

_MODEL_TYPE_MAP = {
    "pi0": _openpi_model.ModelType.PI0,
    "pi05": _openpi_model.ModelType.PI05,
    "pi0_fast": _openpi_model.ModelType.PI0_FAST,
}

_REPACK_KEYS = {
    "libero": {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
    "libero_v2": {
        "observation/image": "observation.images.image",
        "observation/wrist_image": "observation.images.wrist_image",
        "observation/state": "observation.state",
        "actions": "action",
        "prompt": "prompt",
    },
    "franka": {
        "observation/image": "image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
    "franka_co_train": {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
    },
}


@dataclass
class NormStats:
    """Normalization statistics compatible with OpenPI format."""

    mean: np.ndarray
    std: np.ndarray
    q01: Optional[np.ndarray] = None
    q99: Optional[np.ndarray] = None
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None


def _dict_to_norm_stats(data: dict[str, Any]) -> NormStats:
    return NormStats(
        mean=np.array(data["mean"]),
        std=np.array(data["std"]),
        q01=np.array(data["q01"]) if data.get("q01") is not None else None,
        q99=np.array(data["q99"]) if data.get("q99") is not None else None,
        min=np.array(data["min"]) if data.get("min") is not None else None,
        max=np.array(data["max"]) if data.get("max") is not None else None,
    )


def load_stats(norm_stats_path: Path) -> dict[str, NormStats]:
    """Load normalization stats from a JSON file in OpenPI format."""
    if not norm_stats_path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {norm_stats_path}")

    with open(norm_stats_path, "r") as f:
        data = json.load(f)

    if "norm_stats" in data:
        data = data["norm_stats"]

    return {key: _dict_to_norm_stats(stats_dict) for key, stats_dict in data.items()}


class ReturnNormalizer(DataTransformFn):
    """Normalize return values for value model training."""

    def __init__(
        self,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        norm_stats: Optional[dict[str, NormStats]] = None,
        norm_stats_path: Optional[Path] = None,
        return_key: str = "return",
        keep_continuous: bool = True,
        normalize_to_minus_one_zero: bool = True,
    ):
        self.return_key = return_key
        self.keep_continuous = keep_continuous
        self.normalize_to_minus_one_zero = normalize_to_minus_one_zero

        if return_min is not None and return_max is not None:
            self.return_min = return_min
            self.return_max = return_max
        elif norm_stats is not None:
            self._load_from_norm_stats(norm_stats)
        elif norm_stats_path is not None:
            self._load_from_norm_stats(load_stats(Path(norm_stats_path)))
        else:
            raise ValueError(
                "Must provide either (return_min, return_max), norm_stats, "
                "or norm_stats_path"
            )

        logger.info(
            "ReturnNormalizer: return_min=%.4f, return_max=%.4f, range=%s",
            self.return_min,
            self.return_max,
            "(-1, 0)" if self.normalize_to_minus_one_zero else "(0, 1)",
        )

    def _load_from_norm_stats(self, norm_stats: dict[str, NormStats]):
        if "return" not in norm_stats:
            raise ValueError("norm_stats must contain 'return' key")
        rs = norm_stats["return"]
        self.return_min = float(rs.min[0] if hasattr(rs.min, "__len__") else rs.min)
        self.return_max = float(rs.max[0] if hasattr(rs.max, "__len__") else rs.max)

    def normalize_value(self, value: float) -> float:
        if self.normalize_to_minus_one_zero:
            return normalized_return(value, self.return_min, self.return_max)
        span = self.return_max - self.return_min
        if span == 0:
            return 0.0
        return (value - self.return_min) / span

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.return_key not in data:
            return data

        raw = data[self.return_key]
        if isinstance(raw, torch.Tensor):
            raw = raw.item() if raw.numel() == 1 else raw.cpu().numpy()
        elif isinstance(raw, np.ndarray):
            raw = raw.item() if raw.size == 1 else float(raw.flatten()[0])

        result = dict(data)
        result["return_normalized"] = self.normalize_value(float(raw))

        if not self.keep_continuous:
            del result[self.return_key]

        return result


class ValueDataset(Dataset):
    """Flat dataset for value model SFT."""

    def __init__(
        self,
        dataset_path: str,
        robot_type: str,
        model_type: str,
        action_horizon: int = 10,
        action_dim: Optional[int] = None,
        default_prompt: Optional[str] = None,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        normalize_to_minus_one_zero: bool = True,
        max_samples: Optional[int] = None,
        tag: Optional[str] = None,
        episode_percentage: Optional[float] = None,
        shuffle_episodes: bool = False,
        episode_seed: int = 42,
        episode_outcome: Optional[str] = None,
        episode_frame_limits: Optional[Mapping[int, int]] = None,
        **kwargs,
    ):
        _known_unused = {
            "gamma",
            "split",
            "repo_id",
            "norm_stats_dir",
            "asset_id",
            "extra_delta_transform",
            "action_norm_skip_dims",
        }
        unexpected = set(kwargs) - _known_unused
        if unexpected:
            logger.warning(f"ValueDataset ignoring unexpected kwargs: {unexpected}")

        self.max_samples = max_samples
        local_path = Path(dataset_path).absolute()

        self.dataset_meta = LeRobotDatasetMetadata(local_path.name, root=local_path)
        if "action" in self.dataset_meta.features:
            action_key = "action"
        elif "actions" in self.dataset_meta.features:
            action_key = "actions"
        else:
            raise ValueError(
                f"No action key in dataset features: "
                f"{list(self.dataset_meta.features.keys())}"
            )
        delta_timestamps = {
            action_key: [t / self.dataset_meta.fps for t in range(action_horizon)]
        }
        self._base = LeRobotDataset(
            local_path.name,
            root=local_path,
            episodes=sorted(self.dataset_meta.episodes),
            delta_timestamps=None
            if robot_type.lower() in ("robotwin", "aloha")
            else delta_timestamps,
            download_videos=False,
        )
        self._base.hf_dataset.set_transform(decode_image_struct_batch)

        self._sidecar = load_returns_sidecar(local_path, tag)
        if self._sidecar is None:
            raise FileNotFoundError(
                f"Returns sidecar not found for {dataset_path}. "
                f"Run compute_returns.py first to generate "
                f"meta/returns{'_' + tag if tag else ''}.parquet"
            )

        selected_episodes = set(self.dataset_meta.episodes)
        if episode_percentage is not None and episode_percentage < 100:
            if episode_percentage <= 0:
                raise ValueError(
                    f"episode_percentage must be > 0, got {episode_percentage}"
                )
            total = self.dataset_meta.total_episodes
            num = max(1, int(total * episode_percentage / 100.0))
            all_eps = sorted(self.dataset_meta.episodes)
            if shuffle_episodes:
                rng = np.random.default_rng(episode_seed)
                selected_episodes = set(
                    rng.choice(all_eps, size=num, replace=False).tolist()
                )
            else:
                selected_episodes = set(all_eps[:num])

        if episode_outcome is not None:
            if episode_outcome not in ("success", "failure"):
                raise ValueError(
                    "episode_outcome must be 'success', 'failure', or None"
                )
            expected_success = episode_outcome == "success"
            outcomes = {}
            for ep in selected_episodes:
                table = pq.read_table(
                    local_path / self.dataset_meta.get_data_file_path(ep),
                    columns=["is_success"],
                )
                flag = np.asarray(table["is_success"][-1].as_py())
                if flag.size != 1:
                    raise ValueError(
                        f"Episode {ep} must have a scalar terminal is_success flag"
                    )
                outcomes[ep] = bool(flag.item())
            selected_episodes = {
                ep
                for ep, is_success in outcomes.items()
                if is_success == expected_success
            }
            if not selected_episodes:
                raise ValueError(
                    f"No {episode_outcome} episodes found in {dataset_path}"
                )

        self._indices = None
        self.selected_episode_indices = tuple(sorted(selected_episodes))
        all_episodes = set(self.dataset_meta.episodes)
        limits = dict(episode_frame_limits or {})
        if not limits.keys() <= all_episodes or any(v < 0 for v in limits.values()):
            raise ValueError("Invalid critic episode frame limits")
        if selected_episodes != all_episodes or limits:
            idx = self._base.episode_data_index
            self._indices = [
                i
                for position, ep in enumerate(sorted(all_episodes))
                if ep in selected_episodes
                for i in range(
                    idx["from"][position].item(),
                    min(idx["to"][position].item(),
                        idx["from"][position].item() + limits.get(ep, len(self._base))),
                )
            ]

        self._transform = self._build_transform(
            robot_type=robot_type,
            model_type=model_type,
            action_dim=action_dim or 32,
            default_prompt=default_prompt,
        )
        self._tasks = load_task_descriptions(local_path) or (
            self.dataset_meta.tasks if hasattr(self.dataset_meta, "tasks") else None
        )
        self._normalizer = (
            ReturnNormalizer(
                return_min=return_min,
                return_max=return_max,
                normalize_to_minus_one_zero=normalize_to_minus_one_zero,
            )
            if return_min is not None and return_max is not None
            else None
        )

        n = len(self._indices) if self._indices is not None else len(self._base)
        logger.info(f"ValueDataset: {dataset_path}, {min(n, max_samples or n)} samples")

    @staticmethod
    def _build_transform(robot_type, model_type, action_dim, default_prompt):
        if robot_type.lower() in ("robotwin", "aloha"):
            return robotwin_value_input
        model_type_lower = model_type.lower()
        model_type_enum = _MODEL_TYPE_MAP[model_type_lower]
        robot = robot_type.lower()

        transforms_list = []
        repack_keys = _REPACK_KEYS.get(robot)
        if repack_keys is None:
            raise ValueError(
                f"Unknown robot type: {robot_type}. "
                f"Available: {list(_REPACK_KEYS.keys())}"
            )
        transforms_list.append(_openpi_transforms.RepackTransform(repack_keys))

        if robot in ("libero", "libero_v2"):
            transforms_list.append(
                libero_policy.LiberoInputs(model_type=model_type_enum)
            )
        elif robot in ("franka", "franka_co_train"):
            transforms_list.append(
                franka_policy.FrankaEEInputs(
                    action_dim=action_dim,
                    model_type=model_type_enum,
                )
            )

        transforms_list.append(_openpi_transforms.InjectDefaultPrompt(default_prompt))
        transforms_list.append(_openpi_transforms.PadStatesAndActions(action_dim))

        return _openpi_transforms.compose(transforms_list)

    def __len__(self) -> int:
        n = len(self._indices) if self._indices is not None else len(self._base)
        return min(n, self.max_samples) if self.max_samples else n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self._indices[idx] if self._indices is not None else idx
        sample = self._base[real_idx]

        ep = int(sample.get("episode_index", -1))
        fr = int(sample.get("frame_index", -1))
        if ep < 0 or fr < 0:
            raise KeyError(
                f"LeRobot sample missing episode_index ({ep}) or "
                f"frame_index ({fr}) at real_idx={real_idx}. "
                f"Available keys: {sorted(sample.keys())}"
            )

        if self._tasks and "task_index" in sample:
            ti = sample["task_index"]
            ti = ti.item() if isinstance(ti, torch.Tensor) else int(ti)
            if ti in self._tasks:
                sample = {**sample, "prompt": self._tasks[ti]}

        if self._transform is not None:
            sample = self._transform(sample)

        if ep not in self._sidecar:
            raise KeyError(
                f"Episode {ep} not found in returns sidecar at "
                f"real_idx={real_idx}. The sidecar/tag may not match the "
                f"dataset; re-run compute_returns.py with the correct tag."
            )
        raw = float(self._sidecar[ep]["return"][fr])
        target_value = (
            self._normalizer.normalize_value(raw) if self._normalizer else raw
        )

        images = sample.get("image", sample.get("images", {}))
        if not isinstance(images, dict):
            images = {}
        masks = sample.get("image_mask", sample.get("image_masks"))

        result: dict[str, Any] = {
            "images": images,
            "prompt": sample.get("prompt", "perform the task"),
            "target_values": target_value,
            # The sidecar terminal reward encodes the verified episode outcome.
            "episode_success": bool(self._sidecar[ep]["reward"][-1] == 0),
            "actions": None,
        }
        if isinstance(masks, dict) and masks:
            result["image_masks"] = masks
        return result


class ValueDataLoaderImpl(BaseDataLoaderImpl):
    """Lightweight wrapper that yields batches and exposes data_config()."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._data_loader


class ValueMixtureDataset(ReCapMixtureDataset):
    """Mixture of multiple value datasets with weighted sampling."""

    mixture_name = "ValueMixtureDataset"

    def __init__(
        self,
        datasets: Sequence[tuple[Dataset, float] | tuple[Dataset, float, str]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        seed: int = 42,
        group_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Sample configured dataset groups, then frames uniformly per group."""
        if any(len(item) not in (2, 3) for item in datasets):
            raise ValueError("Expected (dataset, weight[, sampling_group]) entries")
        grouped = [len(item) == 3 for item in datasets]
        if any(grouped) and not all(grouped):
            raise ValueError("All value datasets must specify a sampling group")
        if not any(grouped):
            if group_weights is not None:
                raise ValueError("group_weights requires a group on every dataset")
            super().__init__(datasets, mode, balance_dataset_weights, seed)
            return
        if not group_weights:
            raise ValueError("Grouped value sampling requires group_weights")
        if any(not np.isfinite(w) or w <= 0 for w in group_weights.values()):
            raise ValueError("Sampling group weights must be finite and positive")

        group_lengths = {}
        for dataset, weight, group in datasets:
            if weight != 1.0:
                raise ValueError(
                    "Grouped sampling uses group_weights; omit dataset weight"
                )
            if group not in group_weights:
                raise ValueError(f"No configured sampling weight for group {group!r}")
            group_lengths[group] = group_lengths.get(group, 0) + len(dataset)
        if set(group_lengths) != set(group_weights) or any(
            length == 0 for length in group_lengths.values()
        ):
            raise ValueError("Every configured sampling group must contain frames")
        weight_sum = sum(group_weights.values())
        if not np.isclose(weight_sum, 1.0):
            raise ValueError(f"Sampling group weights must sum to 1, got {weight_sum}")

        weighted_datasets = [
            (
                dataset,
                group_weights[group] * len(dataset) / group_lengths[group],
            )
            for dataset, _, group in datasets
        ]
        super().__init__(weighted_datasets, mode, False, seed)
        self.group_weights = dict(group_weights)
