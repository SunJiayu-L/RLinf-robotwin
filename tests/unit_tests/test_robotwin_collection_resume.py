# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Interrupted collection preserves committed episodes and bounded append memory."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pyarrow.parquet as pq
import pytest

from examples.recap.collection_resume import (
    prepare_resume,
    read_summaries,
    resume_writer,
    seek_batch,
)


def test_seek_advances_only_seed_cursor() -> None:
    env = SimpleNamespace(update_reset_state_ids=Mock(), reset=Mock())
    seek_batch(env, 43)
    assert env.update_reset_state_ids.call_count == 43
    env.reset.assert_not_called()


def test_resume_rejects_noncontiguous_split_indices(tmp_path) -> None:
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps({"split": "val", "episode_index": 1}) + "\n"
    )
    with pytest.raises(ValueError, match="Non-contiguous"):
        read_summaries(tmp_path, 4, 2)


def test_interrupted_suffix_is_quarantined_without_touching_committed_data(
    tmp_path,
) -> None:
    directory = tmp_path / "val"
    meta = directory / "meta"
    meta.mkdir(parents=True)
    info = {
        "total_episodes": 2,
        "total_frames": 9,
        "chunks_size": 1000,
        "total_chunks": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (meta / "info.json").write_text(json.dumps(info))
    for name in ["episodes.jsonl", "episodes_stats.jsonl"]:
        (meta / name).write_text(
            "".join(json.dumps({"episode_index": i}) + "\n" for i in range(2))
        )
    data = directory / "data/chunk-000"
    data.mkdir(parents=True)
    original = data / "episode_000000.parquet"
    original.write_bytes(b"committed")
    (data / "episode_000001.parquet").write_bytes(b"uncommitted")
    terminal = tmp_path / "terminal_observations"
    terminal.mkdir()
    (terminal / "val_000000.npz").write_bytes(b"terminal")
    (terminal / "val_000001.npz").write_bytes(b"uncommitted terminal")
    images = directory / "images"
    images.mkdir()
    (images / "partial.png").write_bytes(b"partial")
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps(
            {
                "split": "val",
                "episode_index": 0,
                "frames": 4,
                "terminal_observation": "terminal_observations/val_000000.npz",
            }
        )
        + "\n"
    )

    result = prepare_resume(tmp_path, 4, 2)
    assert result["episodes"] == 1
    assert original.read_bytes() == b"committed"
    assert not images.exists()
    assert not (data / "episode_000001.parquet").exists()
    assert len(list((tmp_path / "recovery").rglob("partial.png"))) == 1
    restored = json.loads((meta / "info.json").read_text())
    assert restored["total_episodes"] == 1 and restored["total_frames"] == 4
    assert len((meta / "episodes_stats.jsonl").read_text().splitlines()) == 1
    assert prepare_resume(tmp_path, 4, 2)["quarantined"] == []


def test_metadata_behind_committed_journal_is_rejected(tmp_path) -> None:
    meta = tmp_path / "val/meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 0}))
    (tmp_path / "episodes.jsonl").write_text(
        json.dumps({"split": "val", "episode_index": 0}) + "\n"
    )
    with pytest.raises(ValueError, match="behind committed"):
        prepare_resume(tmp_path, 4, 2)


def test_append_after_restart_preserves_images_and_global_frame_indices(
    tmp_path,
) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "action": {"dtype": "float32", "shape": (2,), "names": None},
        "observation.images.camera": {
            "dtype": "image",
            "shape": (8, 8, 3),
            "names": ["height", "width", "channels"],
        },
    }
    root = tmp_path / "dataset"
    writer = LeRobotDataset.create(
        repo_id="test_resume",
        root=root,
        fps=15,
        features=features,
        use_videos=False,
    )
    for _ in range(2):
        writer.add_frame(
            {
                "action": np.zeros(2, np.float32),
                "observation.images.camera": np.full((8, 8, 3), 17, np.uint8),
                "task": "first",
            }
        )
    writer.save_episode()
    first_path = root / "data/chunk-000/episode_000000.parquet"
    first_bytes = first_path.read_bytes()
    resumed = resume_writer(root, "test_resume")
    try:
        assert resumed.meta.total_frames == 2
        assert len(resumed.hf_dataset) == 0
        assert resumed.hf_dataset._data.nbytes == 0
        resumed.add_frame(
            {
                "action": np.ones(2, np.float32),
                "observation.images.camera": np.full((8, 8, 3), 31, np.uint8),
                "task": "second",
            }
        )
        resumed.save_episode()
        assert resumed.meta.total_episodes == 2 and resumed.meta.total_frames == 3
        assert first_path.read_bytes() == first_bytes
        table = pq.read_table(root / "data/chunk-000/episode_000001.parquet")
        assert table["index"].to_pylist() == [2]
        assert table["episode_index"].to_pylist() == [1]
        assert table["task_index"].to_pylist() == [1]
        assert table["observation.images.camera"][0].as_py()["bytes"]
    finally:
        resumed.stop_image_writer()


def mock_segment_supervisor(monkeypatch, tmp_path, exit_code, error):
    """Emulate subprocess completion while retaining real supervisor control flow."""
    from examples.recap import recollect_pick_dual2048 as supervisor

    (tmp_path / "logs").mkdir()
    (tmp_path / "datasets/rollout").mkdir(parents=True)
    for rank in range(8):
        (tmp_path / "datasets/rollout" / f"rank_{rank:02d}").mkdir()
    counts = [10]
    limits = []
    monkeypatch.setattr(supervisor, "ROOT", tmp_path)
    monkeypatch.setattr(supervisor, "REPO", tmp_path)
    monkeypatch.setattr(
        supervisor, "prepare_resume", lambda *args: {"episodes": counts[0]}
    )
    monkeypatch.setattr(supervisor, "read_summaries", lambda *args: [{}] * counts[0])
    monkeypatch.setattr(supervisor, "validate", lambda *args: {"episodes": 2048})

    def launch(*args, **kwargs):
        limit = int(kwargs["env"]["RLINF_COLLECT_BATCH_LIMIT"])
        limits.append(limit)
        suffix = kwargs["env"]["RLINF_COLLECT_LOG_SUFFIX"]
        (tmp_path / "logs" / f"collect{suffix}.log").write_text(error)

        def wait():
            if exit_code == 0:
                counts[0] = 12 if len(limits) == 1 else 256
            return exit_code

        return SimpleNamespace(pid=1234, wait=wait)

    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    return supervisor, limits


def test_supervisor_recycles_after_short_resume_probe(monkeypatch, tmp_path):
    supervisor, limits = mock_segment_supervisor(monkeypatch, tmp_path, 0, "")
    assert supervisor.collect_segments(resuming=True) == {"episodes": 2048}
    assert limits == [1, 12]


def test_supervisor_bounds_rendering_retries_without_progress(monkeypatch, tmp_path):
    supervisor, limits = mock_segment_supervisor(
        monkeypatch, tmp_path, 1, "RuntimeError: cannot create buffer"
    )
    with pytest.raises(RuntimeError, match="three attempts without progress"):
        supervisor.collect_segments(resuming=True)
    assert len(limits) == 3


def test_supervisor_does_not_retry_an_unknown_failure(monkeypatch, tmp_path):
    supervisor, limits = mock_segment_supervisor(
        monkeypatch, tmp_path, 1, "ValueError: Partial-batch replay differs"
    )
    with pytest.raises(RuntimeError, match="segment failed"):
        supervisor.collect_segments(resuming=True)
    assert len(limits) == 1
