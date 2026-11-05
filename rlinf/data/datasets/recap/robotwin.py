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

"""Shared RoboTwin observations and RECAP return units.

The critic consumes three RGB views and language. Proprioception remains in the
policy's existing ALOHA transform; no extra state token is invented for the critic.
"""

from __future__ import annotations

import numpy as np
import torch

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
MODEL_CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def normalized_return(value, return_min: float, return_max: float = 0.0):
    """Scale returns and partial reward sums identically; absorbing zero stays zero."""
    if not np.isfinite(return_min) or return_min >= 0 or return_max > 0:
        raise ValueError(
            "RECAP requires a finite negative return_min and return_max <= 0"
        )
    return value / abs(return_min)


def robotwin_value_input(sample: dict) -> dict:
    """Map LeRobot or policy-format RoboTwin observations to critic inputs."""
    images = sample.get("images", sample.get("image", {}))
    mapped = {}
    for camera, model_camera in zip(CAMERAS, MODEL_CAMERAS, strict=True):
        image = images.get(model_camera, images.get(camera))
        if image is None:
            image = sample.get("observation.images." + camera)
        if image is None and "observation/image" in sample:
            if camera == "cam_high":
                image = sample["observation/image"]
            else:
                image = sample["observation/wrist_image"][CAMERAS.index(camera) - 1]
        if image is None:
            raise KeyError(f"Missing RoboTwin RGB camera: {camera}")
        if torch.is_tensor(image):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(
                f"Expected unbatched RGB image for {camera}, got {image.shape}"
            )
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = image.transpose(1, 2, 0)
        if np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"Expected LeRobot video floats in [0,1] for {camera}")
            image = (255 * image).astype(np.uint8)
        if image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"Expected RGB uint8 for {camera}: {image.shape}, {image.dtype}"
            )
        mapped[model_camera] = np.ascontiguousarray(image)
    prompt = sample.get("prompt", sample.get("task"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("RoboTwin RECAP requires the actual nonempty task instruction")
    return {
        "images": mapped,
        "image_masks": dict.fromkeys(MODEL_CAMERAS, True),
        "prompt": prompt,
    }


def per_task_thresholds(
    dataset_results: dict, positive_fraction: float, reference_type: str | None = None
) -> dict:
    """Group by physical task identity, never the instruction's task_index."""
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must be between zero and one")
    grouped = {}
    for result in dataset_results.values():
        task = result["config"].get("task_name")
        if not task:
            raise ValueError("per_task threshold requires task_name for every dataset")
        # Validation episodes must not influence training-label thresholds.
        if not result["config"].get("label_only", False) and (
            reference_type is None or result["config"].get("type") == reference_type
        ):
            grouped.setdefault(task, []).append(result["df"]["advantage"].to_numpy())
    thresholds = {
        task: float(np.quantile(np.concatenate(arrays), 1 - positive_fraction))
        for task, arrays in grouped.items()
    }
    for result in dataset_results.values():
        if result["config"]["task_name"] not in thresholds:
            raise ValueError("No training data for validation task threshold")
    return thresholds
