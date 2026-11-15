# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Collection records native chunks without padded frames or action misalignment."""

import numpy as np
import pytest
import torch

from rlinf.envs.robotwin.collection_utils import append_chunk_frames


def observation(value: int) -> dict:
    """Create distinct camera/state values at one action boundary."""
    return {
        "states": torch.full((3, 14), value, dtype=torch.float64),
        "main_images": torch.full((3, 2, 2, 3), value, dtype=torch.uint8),
        "wrist_images": torch.full((3, 2, 2, 2, 3), value + 10, dtype=torch.uint8),
        "task_descriptions": ["task"] * 3,
    }


def test_pre_action_alignment_early_success_and_inactive_env() -> None:
    frames = [[], [], []]
    actions = np.arange(3 * 4 * 14).reshape(3, 4, 14)
    before = observation(0)
    after = [observation(i + 1) for i in range(4)]
    append_chunk_frames(
        frames, before, actions, after, [2, 4, 0], np.array([True, True, False])
    )
    assert [len(group) for group in frames] == [2, 4, 0]
    for env_index in [0, 1]:
        for action_index, frame in enumerate(frames[env_index]):
            np.testing.assert_array_equal(
                frame["action"], actions[env_index, action_index]
            )
            assert np.all(frame["observation.state"] == action_index)
            assert np.all(frame["observation.images.cam_high"] == action_index)
            assert np.all(
                frame["observation.images.cam_left_wrist"] == action_index + 10
            )
    after[0]["states"].fill_(99)
    actions.fill(99)
    assert frames[0][1]["observation.state"][0] == 1
    assert frames[0][0]["action"][0] == 0


@pytest.mark.parametrize("count", [0, -1, 5])
def test_invalid_execution_count_is_rejected(count: int) -> None:
    with pytest.raises(ValueError, match="executed action count"):
        append_chunk_frames(
            [[], [], []],
            observation(0),
            np.zeros((3, 4, 14)),
            [observation(1)] * 4,
            [count, 0, 0],
            np.array([True, False, False]),
        )


def test_missing_chunk_observations_are_rejected() -> None:
    with pytest.raises(ValueError, match="observation count"):
        append_chunk_frames(
            [[], [], []],
            observation(0),
            np.zeros((3, 4, 14)),
            [observation(1)],
            [4, 0, 0],
            np.array([True, False, False]),
        )
