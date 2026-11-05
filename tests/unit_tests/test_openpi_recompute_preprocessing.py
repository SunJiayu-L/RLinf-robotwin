# Copyright 2025 The RLinf Authors.
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

"""Recomputation must normalize observations after moving to the policy device."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_default_forward_moves_observation_before_normalization():
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/openpi/openpi_action_model.py"
    )
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "OpenPi0ForRLActionPrediction"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "default_forward"
    )
    events = []

    class Observation:
        @classmethod
        def from_dict(cls, data):
            assert data["on_policy_device"], (
                "Image normalization ran before the device transfer"
            )
            events.append("normalize")
            return data

    namespace = {"torch": torch, "_model": SimpleNamespace(Observation=Observation)}
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)

    class Harness:
        default_forward = namespace["default_forward"]
        config = SimpleNamespace(action_chunk=50, action_env_dim=14)

        def input_transform(self, data, transpose=False):
            events.append("transform")
            return {"on_policy_device": False}

        def precision_processor(self, data):
            data["on_policy_device"] = True
            events.append("device")
            return data

        def _preprocess_observation(self, data, train=False):
            events.append("preprocess")
            return (
                [],
                [],
                torch.zeros(2, 200, dtype=torch.long),
                torch.ones(2, 200, dtype=torch.bool),
                torch.zeros(2, 14),
            )

        def get_log_prob_value(self, *args):
            return (
                torch.zeros(2, 1, 50, 14),
                torch.zeros(2, 1),
                torch.zeros(2, 1, 50, 14),
            )

    output = Harness().default_forward(
        {
            "chains": torch.zeros(2, 6, 50, 32),
            "denoise_inds": torch.zeros(2, 5, dtype=torch.long),
        }
    )
    assert events == ["transform", "device", "normalize", "preprocess"]
    assert output["logprobs"].shape == (2, 50, 14)
