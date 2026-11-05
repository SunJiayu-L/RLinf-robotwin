# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the larger, memory-bounded RECAP data experiment."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import spearmanr

from rlinf.algorithms.recap_value_metrics import spearman, value_metrics
from rlinf.envs.robotwin.collection_utils import (
    make_fresh_reset,
    release_saved_arrow_cache,
)


@pytest.mark.parametrize(
    "a,b",
    [
        ([1, 1, 2, 3], [1, 2, 2, 3]),
        ([4, 3, 2, 1], [1, 2, 3, 4]),
        ([2, 2, 1, 3, 3], [3, 2, 2, 1, 1]),
        ([1, 2, 3], [10, 20, 30]),
    ],
)
def test_spearman_matches_standard_with_ties(a, b):
    assert spearman(np.array(a), np.array(b)) == pytest.approx(
        spearmanr(a, b).statistic
    )


def test_constant_predictions_are_undefined_not_perfect():
    result = value_metrics(np.ones(4), np.arange(4))
    assert np.isnan(result["value_spearman"])
    assert result["value_spearman_defined"] == 0


def test_global_ranking_detects_cross_episode_inversion():
    predictions = np.array([-0.5, -0.4, -0.3, -1, -0.9, -0.8])
    targets = np.array([-1, -0.9, -0.8, -0.3, -0.2, -0.1])
    assert spearman(predictions[:3], targets[:3]) == pytest.approx(1)
    assert spearman(predictions[3:], targets[3:]) == pytest.approx(1)
    assert spearman(predictions, targets) < 0


@pytest.mark.parametrize("provided", [None, "Fixed paired evaluation text."])
def test_description_uses_new_scene_and_preserves_explicit_instruction(provided):
    task = SimpleNamespace(ep_num=12)
    task.get_info = lambda: {"scene_seed": task.ep_num}
    task.set_instruction = lambda text: setattr(task, "instruction", text)
    env = SimpleNamespace(
        global_lock=threading.Lock(),
        lock=threading.Lock(),
        task=task,
        args={},
        env_seed=12,
        episode_info_list=[{"scene_seed": 12}],
    )
    env.create_instruction = lambda: (
        f"instruction for scene {env.episode_info_list[0]['scene_seed']}"
    )

    def native_reset(self, env_seed=None, instruction=None):
        self.task.ep_num = env_seed + 1  # Simulate the native unstable-scene fallback.

    reset = make_fresh_reset(native_reset)
    reset(env, env_seed=120, instruction=provided)
    assert env.env_seed == 121
    assert env.task.instruction == (provided or "instruction for scene 121")
    assert env.episode_info_list == [{"scene_seed": 121}]


def test_writer_releases_saved_images_without_losing_disk_data(tmp_path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.images.cam_high": {
            "dtype": "image",
            "shape": (8, 8, 3),
            "names": ["height", "width", "channels"],
        },
    }
    writer = LeRobotDataset.create(
        "bounded_test",
        15,
        root=tmp_path / "episodes",
        robot_type="aloha",
        features=features,
        use_videos=False,
    )
    for ep in range(3):
        for frame in range(2):
            writer.add_frame(
                {
                    "task": f"episode {ep}",
                    "action": np.full(14, ep, np.float32),
                    "observation.state": np.zeros(14, np.float32),
                    "observation.images.cam_high": np.full(
                        (8, 8, 3), ep * 70, np.uint8
                    ),
                }
            )
        writer.save_episode()
        release_saved_arrow_cache(writer)
        assert writer.meta.total_episodes == ep + 1
        assert len(writer.hf_dataset) == 0 and writer.hf_dataset._data.nbytes == 0
    reader = LeRobotDataset(
        "bounded_test", root=tmp_path / "episodes", download_videos=False
    )
    assert len(reader) == 6
    assert reader[4]["action"][0].item() == 2
    assert reader[4]["observation.images.cam_high"][0, 0, 0].item() == pytest.approx(
        140 / 255
    )


def test_new_gate_rejects_hidden_failure_errors(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "examples/recap")
    )
    from run_robotwin4096_pipeline import qualified

    metrics = {
        "rollout/continuous_mae": 0.1,
        "rollout/value_spearman": 0.5,
        "rollout/success_mae": 0.08,
        "rollout/failure_mae": 0.15,
        "rollout/value_spearman_defined": 1.0,
        "datasets_sft_val/continuous_mae": 0.01,
    }
    assert qualified(metrics)
    assert not qualified({**metrics, "rollout/failure_mae": 0.4})
    assert not qualified({**metrics, "rollout/value_spearman": float("nan")})
    assert not qualified({**metrics, "rollout/value_spearman_defined": 0.0})


def test_seed_pool_provides_full_nonoverlapping_train_and_validation():
    import torch

    from rlinf.envs.robotwin.seed_utils import partition_success_seeds

    seeds = torch.arange(4352) * 1024 + 1_000_000
    train, val = [], []
    for rank in range(8):
        shard = partition_success_seeds(
            seeds, base_seed=0, seed_offset=rank, total_num_processes=8, num_group=2
        ).tolist()
        assert len(shard) == 544
        val += shard[:32]
        train += shard[32:]
    assert len(train) == 4096 and len(val) == 256
    assert len(set(train + val)) == 4352


def test_following_collection_stops_after_critic_before_policy(monkeypatch, tmp_path):
    import json
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "examples/recap")
    )
    import run_robotwin4096_pipeline as pipeline

    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["pipeline", "--follow-collection"])
    (tmp_path / "collection_handoff.json").write_text(json.dumps({"completed": []}))
    (tmp_path / "status.tsv").write_text("now\tEND\tcollect\texit=0\n")
    (tmp_path / "execution_plan.json").write_text(json.dumps({"stop_after": "value"}))
    metrics = {
        "rollout/continuous_mae": 0.1,
        "rollout/value_spearman": 0.5,
        "rollout/success_mae": 0.08,
        "rollout/failure_mae": 0.15,
        "rollout/value_spearman_defined": 1.0,
        "datasets_sft_val/continuous_mae": 0.01,
    }
    directory = tmp_path / "runs/value"
    directory.mkdir(parents=True)
    (directory / "value_validation.jsonl").write_text(
        json.dumps({"completed_steps": 500, "metrics": metrics}) + "\n"
    )
    monkeypatch.setattr(
        pipeline,
        "validate",
        lambda *args: {"training_episodes": 4096, "validation_episodes": 256},
    )
    monkeypatch.setattr(
        pipeline, "complete_checkpoint", lambda *args: tmp_path / "actor"
    )
    stages = []
    monkeypatch.setattr(
        pipeline, "run", lambda stage, state, **kwargs: stages.append(stage)
    )
    pipeline.main()
    assert stages == ["returns", "value"]
    state = json.loads((tmp_path / "pipeline_status.json").read_text())
    assert state["status"] == "PAUSED_AFTER_CRITIC"
