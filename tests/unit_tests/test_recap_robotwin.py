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

"""RECAP contracts: observation parity, reward units, task thresholds, and weights."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rlinf.data.datasets.recap.robotwin import (
    normalized_return,
    per_task_thresholds,
    robotwin_value_input,
)
from rlinf.models.embodiment.openpi_cfg.checkpoint import (
    load_pt_policy,
    resolve_pt_checkpoint,
)


def sample():
    return {
        "prompt": "Hit the block with the hammer.",
        **{
            f"observation.images.{key}": np.full((3, 16, 20), value, np.uint8)
            for key, value in zip(
                ["cam_high", "cam_left_wrist", "cam_right_wrist"],
                [13, 51, 221],
                strict=True,
            )
        },
    }


def test_training_and_inference_camera_order_and_rgb_are_identical():
    raw = sample()
    train = robotwin_value_input(raw)
    infer = robotwin_value_input(
        {
            "observation/image": raw["observation.images.cam_high"].transpose(1, 2, 0),
            "observation/wrist_image": np.stack(
                [
                    raw["observation.images.cam_left_wrist"],
                    raw["observation.images.cam_right_wrist"],
                ]
            ).transpose(0, 2, 3, 1),
            "prompt": raw["prompt"],
        }
    )
    for key in train["images"]:
        np.testing.assert_array_equal(train["images"][key], infer["images"][key])
        assert train["images"][key].shape == (16, 20, 3)


def test_no_silent_missing_cameras_or_empty_instructions():
    raw = sample()
    del raw["observation.images.cam_right_wrist"]
    with pytest.raises(KeyError):
        robotwin_value_input(raw)
    with pytest.raises(ValueError, match="instruction"):
        robotwin_value_input({**sample(), "prompt": ""})


def test_video_float_decoding_keeps_three_views():
    raw = sample()
    for key in list(raw):
        if "images" in key:
            raw[key] = raw[key].astype(np.float32) / 255
    result = robotwin_value_input(raw)
    assert len(result["images"]) == 3
    assert all(image.dtype == np.uint8 for image in result["images"].values())


def test_false_vector_success_label_is_not_treated_as_true(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.recap.process.compute_returns import _process_single_parquet

    data = pa.table(
        {
            "episode_index": [0, 0, 0],
            "frame_index": [0, 1, 2],
            "is_success": [[False], [False], [False]],
            "task_index": [0, 0, 0],
        }
    )
    file = tmp_path / "episode.parquet"
    pq.write_table(data, file)
    result = _process_single_parquet(str(file), "rollout", 1.0, -400.0, {0: "hammer"})
    np.testing.assert_allclose(result["return"].to_numpy(), [-402.0, -401.0, -400.0])


@pytest.mark.parametrize("maximum", [0.0, -10.0])
def test_reward_scaling_preserves_bellman_identity_and_terminal_zero(maximum):
    # Perfect values on a deterministic successful episode must have zero advantage.
    returns = np.arange(-99.0, 1.0)
    values = normalized_return(returns, -200.0, maximum)
    for time in [0, 10, 80, 99]:
        lookahead = min(20, len(returns) - time)
        next_value = values[time + lookahead] if time + lookahead < len(values) else 0.0
        reward_sum = returns[time] - (
            returns[time + lookahead] if time + lookahead < len(values) else 0.0
        )
        assert (
            abs(
                normalized_return(reward_sum, -200.0, maximum)
                + next_value
                - values[time]
            )
            < 1e-10
        )
    assert normalized_return(0.0, -200.0, maximum) == 0.0


def test_threshold_groups_use_physical_task_and_exclude_validation():
    def result(task, values, **kwargs):
        return {
            "config": {"task_name": task, **kwargs},
            "df": pd.DataFrame({"advantage": values}),
        }

    results = {
        "demo": result("hammer", [0.1, 0.2]),
        "rollout": result("hammer", [0.3, 0.4]),
        "other": result("phone", [-0.4, -0.3, -0.2, -0.1]),
        "heldout": result("hammer", [100.0, 200.0], label_only=True),
    }
    thresholds = per_task_thresholds(results, 0.25)
    assert thresholds["hammer"] == pytest.approx(0.325)
    assert thresholds["phone"] == pytest.approx(-0.175)
    with pytest.raises(ValueError, match="task_name"):
        per_task_thresholds(
            {"missing": {"config": {}, "df": results["demo"]["df"]}}, 0.4
        )


def test_finetuned_pt_takes_precedence_over_base_safetensors(tmp_path: Path):
    source = torch.nn.Linear(3, 2)
    torch.save(source.state_dict(), tmp_path / "full_weights.pt")
    (tmp_path / "model.safetensors").write_bytes(b"base must never be read")
    assert resolve_pt_checkpoint(str(tmp_path)) == tmp_path / "full_weights.pt"
    target = torch.nn.Linear(3, 2)
    provenance = load_pt_policy(target, resolve_pt_checkpoint(str(tmp_path)))
    assert not provenance["missing_keys"] and not provenance["unexpected_keys"]
    for a, b in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        load_pt_policy(torch.nn.Linear(4, 2), tmp_path / "full_weights.pt")


def test_cls_indices_follow_observation_device_not_sharded_parameter_device():
    from rlinf.models.embodiment.value_model.modeling_critic import ValueHead

    class FSDPEmbeddingStub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(1, 4, device="meta"))

        def forward(self, tokens):
            assert tokens.device.type == "cpu"
            return torch.zeros(*tokens.shape, 4, device=tokens.device)

    head = ValueHead(hidden_size=4, num_bins=201, v_min=-1.0, v_max=0.0)
    head.cls_embedding = FSDPEmbeddingStub()
    assert head.get_cls_embedding(2, device=torch.device("cpu")).shape == (2, 1, 4)


def test_sft_advantage_labels_are_not_forced_positive(tmp_path):
    from examples.recap.process.compute_advantages import save_advantages_to_dataset

    frame = pd.DataFrame(
        {"episode_index": [0, 0], "frame_index": [0, 1], "advantage": [-0.5, 0.5]}
    )
    save_advantages_to_dataset(
        tmp_path, frame, 0.0, dataset_type="sft", tag="test", force_sft_positive=False
    )
    written = pd.read_parquet(tmp_path / "meta/advantages_test.parquet")
    assert written["advantage"].tolist() == [False, True]
    save_advantages_to_dataset(
        tmp_path, frame, 0.0, dataset_type="sft", tag="expert", force_sft_positive=True
    )
    expert = pd.read_parquet(tmp_path / "meta/advantages_expert.parquet")
    assert expert["advantage"].tolist() == [True, True]


def test_cfg_prompts_share_normalized_unpadded_state():
    from rlinf.models.embodiment.openpi_cfg.prompt_transforms import (
        transform_with_guidance,
    )

    observed = []

    def normalize(data):
        return {**data, "state": data["state"] / 10}

    def tokenize(data):
        prompt = data.pop("prompt")
        state = data["state"]
        observed.append((prompt, state.copy()))
        return {
            **data,
            "tokenized_prompt": np.array([len(prompt), state[0]]),
            "tokenized_prompt_mask": np.array([True, True]),
        }

    def pad(data):
        return {**data, "state": np.pad(data["state"], (0, 18))}

    result = transform_with_guidance(
        {"state": np.arange(14, dtype=np.float32) + 10, "prompt": "task"},
        [normalize, tokenize, pad],
        tokenize,
        "task positive",
        "task negative",
    )
    assert len(observed) == 3
    assert {prompt for prompt, state in observed} == {
        "task",
        "task positive",
        "task negative",
    }
    for _, state in observed:
        np.testing.assert_array_equal(
            state, (np.arange(14, dtype=np.float32) + 10) / 10
        )
    assert result["state"].shape == (32,)
    assert result["tokenized_positive_guidance_prompt"][1] == 1
    assert result["tokenized_negative_guidance_prompt"][1] == 1


def test_collector_consumes_manifest_from_first_to_last_without_overrun():
    from examples.recap.collect_robotwin import reset_collection_batch
    from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv

    env = RoboTwinEnv.__new__(RoboTwinEnv)
    env.num_envs = 1
    env.episode_manifest = [
        {"seed": 1000 + index, "instruction": f"instruction {index}"}
        for index in range(12)
    ]
    env._current_episode_index = 0
    env._update_manifest_episodes()
    env.update_reset_state_ids = env._update_manifest_episodes
    env.reset = lambda: ((int(env.reset_state_ids[0]), env.current_instructions[0]), {})
    actual = [reset_collection_batch(env, batch)[0] for batch in range(12)]
    assert actual == [(1000 + index, f"instruction {index}") for index in range(12)]
    with pytest.raises(RuntimeError, match="manifest exhausted"):
        reset_collection_batch(env, 12)
