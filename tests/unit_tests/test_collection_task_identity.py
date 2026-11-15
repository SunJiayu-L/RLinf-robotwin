"""Collection shards must record the actual configured task."""

import pytest

from rlinf.envs.robotwin.collection_utils import collection_task_name


@pytest.mark.parametrize("task", ["beat_block_hammer", "place_phone_stand", "lift_pot"])
def test_collection_task_identity(task):
    assert collection_task_name(
        {"task_config": {"task_name": task}, "task_names": [task]}
    ) == task


def test_collection_rejects_mismatched_task_names():
    with pytest.raises(ValueError, match="single consistent task"):
        collection_task_name(
            {"task_config": {"task_name": "lift_pot"}, "task_names": ["move_can_pot"]}
        )


def test_collection_rejects_mixed_task_shards():
    with pytest.raises(ValueError, match="single consistent task"):
        collection_task_name(
            {
                "task_config": {"task_name": "lift_pot"},
                "task_names": ["lift_pot", "move_can_pot"],
            }
        )
