# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Round-trip a failed LeRobot trajectory through returns and critic input."""

import numpy as np


def test_failed_episode_roundtrip_keeps_images_actions_and_failure(tmp_path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from examples.recap.process.compute_returns import process_dataset
    from rlinf.data.datasets.recap.value_model import ValueDataset

    features = {
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "is_success": {"dtype": "bool", "shape": (1,), "names": None},
        **{
            f"observation.images.{cam}": {
                "dtype": "image",
                "shape": (16, 20, 3),
                "names": ["height", "width", "channels"],
            }
            for cam in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        },
    }
    root = tmp_path / "experience"
    writer = LeRobotDataset.create(
        "recap_test",
        15,
        root=root,
        robot_type="aloha",
        features=features,
        use_videos=False,
    )
    for frame in range(3):
        writer.add_frame(
            {
                "task": "Strike the block.",
                "action": np.full(14, frame, np.float32),
                "observation.state": np.zeros(14, np.float32),
                "is_success": np.array([False]),
                **{
                    f"observation.images.{cam}": np.full((16, 20, 3), value, np.uint8)
                    for cam, value in zip(
                        ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                        [41, 113, 237],
                        strict=True,
                    )
                },
            }
        )
    writer.save_episode()
    process_dataset(root, None, "rollout", 1.0, -400.0, num_workers=1, tag="test")
    reader = ValueDataset(
        str(root),
        "robotwin",
        "pi05",
        action_dim=14,
        return_min=-800.0,
        return_max=0.0,
        tag="test",
    )
    assert len(reader) == 3
    np.testing.assert_allclose(reader[0]["target_values"], -402 / 800)
    np.testing.assert_allclose(reader[2]["target_values"], -0.5)
    assert reader[0]["images"]["right_wrist_0_rgb"][0, 0, 0] == 237
    assert reader[0]["prompt"] == "Strike the block."
