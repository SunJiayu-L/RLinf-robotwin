# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify outcome views preserve returns and success pools ignore data source."""

import numpy as np
import pytest

from rlinf.data.datasets.recap.value_model import ValueDataset, ValueMixtureDataset


class Frames:
    """Minimal frame dataset for testing sampling probabilities."""

    def __init__(self, count: int):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return index


def test_group_sampling_is_uniform_per_frame_across_sources_and_shards():
    expert, success_a, success_b, failure = map(Frames, [10, 15, 25, 100])
    mixture = ValueMixtureDataset(
        [
            (expert, 1.0, "success"),
            (success_a, 1.0, "success"),
            (success_b, 1.0, "success"),
            (failure, 1.0, "failure"),
        ],
        group_weights={"success": 0.65, "failure": 0.35},
    )
    probabilities = mixture.dataset_sampling_weights
    np.testing.assert_allclose(probabilities, [0.13, 0.195, 0.325, 0.35])
    np.testing.assert_allclose(probabilities[:3] / [10, 15, 25], 0.65 / 50)
    counts = np.zeros(4)
    for index in range(10000):
        dataset, frame = mixture._sample_step(index)
        assert 0 <= frame < len(dataset)
        counts[mixture.datasets.index(dataset)] += 1
    np.testing.assert_allclose(counts / counts.sum(), probabilities, atol=0.012)


@pytest.mark.parametrize(
    "entries,weights",
    [
        ([(Frames(1), 1.0, "success")], {"failure": 1.0}),
        ([(Frames(0), 1.0, "success")], {"success": 1.0}),
        ([(Frames(1), 1.0, "success")], {"success": -1.0}),
        ([(Frames(1), 0.5, "success")], {"success": 1.0}),
        ([(Frames(1), 1.0)], {"success": 1.0}),
    ],
)
def test_invalid_pool_configuration_is_rejected(entries, weights):
    with pytest.raises(ValueError):
        ValueMixtureDataset(entries, group_weights=weights)


def test_legacy_mixture_keeps_explicit_dataset_weights():
    mixture = ValueMixtureDataset(
        [(Frames(10), 0.5), (Frames(100), 0.5)], balance_dataset_weights=False
    )
    np.testing.assert_allclose(mixture.dataset_sampling_weights, [0.5, 0.5])


def test_outcome_views_use_terminal_flag_and_preserve_frame_return_alignment(tmp_path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from examples.recap.process.compute_returns import process_dataset

    root = tmp_path / "experience"
    features = {
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        "is_success": {"dtype": "bool", "shape": (1,), "names": None},
        **{
            f"observation.images.{camera}": {
                "dtype": "image",
                "shape": (16, 20, 3),
                "names": ["height", "width", "channels"],
            }
            for camera in ["cam_high", "cam_left_wrist", "cam_right_wrist"]
        },
    }
    writer = LeRobotDataset.create(
        "outcome_test", 15, root=root, features=features, use_videos=False
    )
    for episode, length in enumerate([2, 4]):
        for frame in range(length):
            writer.add_frame(
                {
                    "task": "Pick both bottles.",
                    "action": np.zeros(14, dtype=np.float32),
                    "is_success": np.array([episode == 0 and frame == length - 1]),
                    **{
                        f"observation.images.{camera}": np.full(
                            (16, 20, 3), 20 + episode * 50 + frame, dtype=np.uint8
                        )
                        for camera in [
                            "cam_high",
                            "cam_left_wrist",
                            "cam_right_wrist",
                        ]
                    },
                }
            )
        writer.save_episode()
    process_dataset(root, None, "rollout", 1.0, -400, num_workers=1, tag="test")
    options = {
        "dataset_path": str(root),
        "robot_type": "robotwin",
        "model_type": "pi05",
        "return_min": -800,
        "return_max": 0,
        "tag": "test",
    }
    full = ValueDataset(**options)
    success = ValueDataset(**options, episode_outcome="success")
    failure = ValueDataset(**options, episode_outcome="failure")
    assert len(full) == 6
    assert len(success) == 2
    assert len(failure) == 4
    assert success.selected_episode_indices == (0,)
    assert failure.selected_episode_indices == (1,)
    np.testing.assert_allclose(
        [success[i]["target_values"] for i in range(2)], [-1 / 800, 0]
    )
    np.testing.assert_allclose(
        [failure[i]["target_values"] for i in range(4)],
        np.arange(-403, -399) / 800,
    )
    assert failure[0]["images"]["base_0_rgb"][0, 0, 0] == 70
    assert "is_success" not in success[0]


def test_target_validation_preserves_original_expert_episode_ids(tmp_path, monkeypatch):
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf

    from examples.recap import train_pick_dual_chunk_critic as supervisor

    (tmp_path / "protocol").mkdir()
    dataset = tmp_path / "expert"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta/info.json").write_text(json.dumps({"total_episodes": 1}))
    (dataset / "meta/episodes.jsonl").write_text(
        json.dumps({"episode_index": 4500, "length": 2}) + "\n"
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": [4500, 4500],
                "frame_index": [0, 1],
                "reward": [-1.0, 0.0],
                "return": [-1.0, 0.0],
                "prompt": ["Pick both bottles."] * 2,
            }
        ),
        dataset / "meta/returns_test.parquet",
    )
    OmegaConf.save(
        OmegaConf.create(
            {
                "data": {
                    "tag": "test",
                    "train_data_paths": [{"dataset_path": str(dataset), "type": "sft"}],
                }
            }
        ),
        tmp_path / "protocol/returns.yaml",
    )
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    assert supervisor.validate_targets()["status"] == "PASS"


def test_resource_queue_waits_for_two_ready_checks_and_honors_pause(
    tmp_path, monkeypatch
):
    from examples.recap import train_pick_dual_chunk_critic as supervisor

    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    states = iter([False, True, False, True, True])
    waits = []
    monkeypatch.setattr(supervisor, "resource_status", lambda: {"ready": next(states)})
    monkeypatch.setattr(supervisor.time, "sleep", lambda seconds: waits.append(seconds))
    assert supervisor.wait_for_resources()
    assert len(waits) == 4
    (tmp_path / "USER_PAUSED").touch()
    assert not supervisor.wait_for_resources()
