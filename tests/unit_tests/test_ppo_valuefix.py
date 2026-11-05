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

"""CPU regression tests for PPO value semantics, pooled statistics, and RNG."""

import ast
import importlib.util
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stats_module = load_file("value_stats_tested", ROOT / "rlinf/algorithms/value_stats.py")
rng_module = load_file("evaluation_rng_tested", ROOT / "rlinf/utils/evaluation_rng.py")


def live_methods():
    file = ROOT / "rlinf/models/embodiment/openpi/openpi_action_model.py"
    tree = ast.parse(file.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OpenPi0ForRLActionPrediction"
    )
    namespace = {"torch": torch, "random": random}
    for name in ["sample_actions", "get_log_prob_value"]:
        node = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        node.decorator_list = []
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                node,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(file), "exec"), namespace)
    return namespace


METHODS = live_methods()


class Harness:
    sample_actions = METHODS["sample_actions"]
    get_log_prob_value = METHODS["get_log_prob_value"]
    use_vlm_value = False

    def __init__(self, joint=False):
        self.config = SimpleNamespace(
            num_steps=5,
            action_horizon=50,
            action_dim=14,
            joint_logprob=joint,
            is_nft=False,
            ignore_last=False,
            noise_method="flow_sde",
            action_chunk=50,
            action_env_dim=14,
        )

    def _preprocess_observation(self, obs, train=False):
        return [], [], None, None, obs.state

    def _build_prefix_cache(self, *args):
        return None, None, None

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def _init_nft_state(self, *args):
        return {}

    def _update_nft_state(self, *args):
        pass

    def sample_mean_var_val(self, x, idx, *args):
        value = (
            torch.full((x.shape[0],), float(idx))
            if isinstance(idx, int)
            else idx.float()
        )
        return x, torch.zeros_like(x), value, x

    def get_logprob_norm(self, sample, *args):
        return torch.zeros_like(sample)

    def gaussian_entropy(self, sigma):
        return torch.zeros_like(sigma)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("joint", [False, True])
def test_value_recompute_matches_rollout_before_update(seed, joint):
    model = Harness(joint)
    observation = SimpleNamespace(state=torch.zeros(2, 14))
    random.seed(seed)
    result = model.sample_actions(observation, compute_values=True, mode="train")
    _, values, _ = model.get_log_prob_value(
        [],
        [],
        None,
        None,
        observation.state,
        result["chains"],
        result["denoise_inds"],
        True,
    )
    torch.testing.assert_close(result["prev_values"].flatten(), values.mean(-1))


def test_eval_minus_one_denoise_index_remains_supported():
    model = Harness()
    result = model.sample_actions(
        SimpleNamespace(state=torch.zeros(2, 14)), compute_values=True, mode="eval"
    )
    assert result["prev_values"].shape == (2, 1)
    assert torch.isfinite(result["prev_values"]).all()


def test_pooled_statistics_ignore_padding_and_match_full_dataset():
    y = torch.tensor([0.0, 1.0, 2.0, 100.0])
    v = torch.tensor([0.1, 0.8, 1.7, -100.0])
    mask = torch.tensor([True, True, True, False])
    s = stats_module.value_sufficient_statistics(v, y, mask)
    parts = sum(
        (
            stats_module.value_sufficient_statistics(
                v[i : i + 1], y[i : i + 1], mask[i : i + 1]
            )
            for i in range(4)
        )
    )
    torch.testing.assert_close(s, parts)
    metrics = stats_module.value_metrics_from_statistics(parts)
    expected = 1 - torch.var((y - v)[mask].double(), correction=0) / torch.var(
        y[mask].double(), correction=0
    )
    torch.testing.assert_close(metrics["critic/explained_variance"], expected)
    assert metrics["critic/valid_samples"] == 3
    torch.testing.assert_close(
        metrics["critic/mse"], ((y - v)[mask].double().square()).mean()
    )


@pytest.mark.parametrize("size", [0, 1, 2])
def test_undefined_microbatch_ev_does_not_poison_pooled_ev(size):
    constant = torch.ones(size)
    small = stats_module.value_sufficient_statistics(constant, constant)
    assert math.isnan(
        stats_module.value_metrics_from_statistics(small)["critic/explained_variance"]
    )
    extra = stats_module.value_sufficient_statistics(
        torch.tensor([0.0, 2.0]), torch.tensor([0.0, 2.0])
    )
    assert (
        stats_module.value_metrics_from_statistics(small + extra)[
            "critic/explained_variance"
        ]
        == 1
    )


def test_evaluation_rng_repeats_and_restores_training_generators():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    draws = []
    for _ in range(2):
        with rng_module.evaluation_rng(777, 0, "cpu"):
            draws.append((random.random(), np.random.rand(), torch.rand(1)))
    assert draws[0][:2] == draws[1][:2]
    torch.testing.assert_close(draws[0][2], draws[1][2])
    assert (random.random(), np.random.rand()) == expected[:2]
    torch.testing.assert_close(torch.rand(1), expected[2])


def test_clip_ratio_uses_unclipped_values_and_valid_samples():
    # Extract the actual critic loss with lightweight helper dependencies.
    file = ROOT / "rlinf/algorithms/losses.py"
    tree = ast.parse(file.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_ppo_critic_loss"
    )
    namespace = {"torch": torch, "Optional": __import__("typing").Optional}
    namespace["huber_loss"] = lambda error, delta: torch.where(
        error.abs() < delta, 0.5 * error.square(), delta * (error.abs() - 0.5 * delta)
    )
    namespace["masked_mean"] = lambda value, mask, *args: (
        value[mask].mean() if mask.any() else (value * mask).sum()
    )
    namespace["masked_mean_ratio"] = lambda value, mask, ratio: (
        value / ratio * mask
    ).mean()
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            str(file),
            "exec",
        ),
        namespace,
    )
    _, metrics = namespace["compute_ppo_critic_loss"](
        values=torch.tensor([1.0, 0.0]),
        returns=torch.tensor([1.0, 0.0]),
        prev_values=torch.tensor([0.0, 0.0]),
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=torch.tensor([True, False]),
    )
    assert metrics["critic/value_clip_ratio"] == 1
