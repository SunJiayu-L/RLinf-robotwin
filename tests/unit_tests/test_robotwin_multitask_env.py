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

import json
import sys
import types

import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv


class FakeVectorEnv:
    instances = []

    def __init__(
        self,
        task_config,
        n_envs,
        env_seeds=None,
        instructions=None,
        instruction_type="seen",
        strict_episode_manifest=False,
    ):
        self.task_config = task_config
        self.task_name = task_config["task_name"]
        self.n_envs = n_envs
        self.env_seeds = list(env_seeds or range(n_envs))
        self.instructions = (
            list(instructions)
            if instructions is not None
            else [f"{self.task_name}-instruction-{idx}" for idx in range(n_envs)]
        )
        self.reset_calls = []
        self.step_calls = []
        self.instruction_type = instruction_type
        self.strict_episode_manifest = strict_episode_manifest
        FakeVectorEnv.instances.append(self)

    def reset(self, env_idx=None, env_seeds=None, instructions=None):
        if env_idx is None:
            indices = list(range(self.n_envs))
        elif isinstance(env_idx, list):
            indices = env_idx
        else:
            indices = [env_idx]
        self.reset_calls.append(
            {
                "env_idx": indices,
                "env_seeds": list(env_seeds) if env_seeds is not None else None,
                "instructions": list(instructions) if instructions is not None else None,
            }
        )
        if env_seeds is not None:
            for local_idx, seed in zip(indices, env_seeds, strict=True):
                self.env_seeds[local_idx] = int(seed)
        if instructions is not None:
            for local_idx, instruction in zip(indices, instructions, strict=True):
                self.instructions[local_idx] = instruction

    def step(self, actions):
        self.step_calls.append(np.array(actions).copy())
        rewards = [float(seed) for seed in self.env_seeds]
        terminations = [False] * self.n_envs
        truncations = [False] * self.n_envs
        infos = [{"success": False, "task": self.task_name} for _ in range(self.n_envs)]
        return self.get_obs(), rewards, terminations, truncations, infos

    def get_obs(self):
        return [
            {
                "full_image": np.zeros((2, 2, 3), dtype=np.uint8),
                "left_wrist_image": None,
                "right_wrist_image": None,
                "state": np.full((3,), idx, dtype=np.float32),
                "instruction": self.instructions[idx],
            }
            for idx in range(self.n_envs)
        ]

    def close(self, clear_cache=True):
        return None


def _install_fake_robotwin(monkeypatch):
    FakeVectorEnv.instances = []
    robotwin_module = types.ModuleType("robotwin")
    envs_module = types.ModuleType("robotwin.envs")
    vector_env_module = types.ModuleType("robotwin.envs.vector_env")
    vector_env_module.VectorEnv = FakeVectorEnv
    monkeypatch.setitem(sys.modules, "robotwin", robotwin_module)
    monkeypatch.setitem(sys.modules, "robotwin.envs", envs_module)
    monkeypatch.setitem(sys.modules, "robotwin.envs.vector_env", vector_env_module)


def _make_cfg(seeds_path):
    return OmegaConf.create(
        {
            "seed": 0,
            "auto_reset": False,
            "use_rel_reward": True,
            "ignore_terminations": False,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "use_custom_reward": False,
            "video_cfg": {},
            "task_names": ["task_a", "task_b"],
            "center_crop": False,
            "assets_path": "/tmp/robotwin-assets",
            "seeds_path": str(seeds_path),
            "reward_coef": 1.0,
            "max_episode_steps": 10,
            "task_config": {
                "task_name": "task_a",
                "save_path": "/tmp/robotwin-save",
                "step_lim": 10,
            },
        }
    )


def _write_seed_file(path):
    path.write_text(
        json.dumps(
            {
                "task_a": {"success_seeds": list(range(1000, 1020))},
                "task_b": {"success_seeds": list(range(2000, 2020))},
            }
        )
    )


def test_robotwin_multitask_builds_one_vector_env_per_task(monkeypatch, tmp_path):
    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "train_seeds.json"
    _write_seed_file(seeds_path)

    env = RoboTwinEnv(
        _make_cfg(seeds_path),
        num_envs=4,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    assert env.is_multitask
    assert env.env_task_names == ["task_a", "task_b", "task_a", "task_b"]
    assert [venv.task_name for venv in FakeVectorEnv.instances] == ["task_a", "task_b"]
    assert [venv.n_envs for venv in FakeVectorEnv.instances] == [2, 2]
    assert all(1000 <= int(env.reset_state_ids[idx]) < 1020 for idx in [0, 2])
    assert all(2000 <= int(env.reset_state_ids[idx]) < 2020 for idx in [1, 3])


def test_robotwin_multitask_reset_maps_global_to_local_indices(monkeypatch, tmp_path):
    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "train_seeds.json"
    _write_seed_file(seeds_path)
    env = RoboTwinEnv(
        _make_cfg(seeds_path),
        num_envs=4,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )

    env.reset(env_idx=[1, 2])

    task_a_venv = env.venvs["task_a"]
    task_b_venv = env.venvs["task_b"]
    assert task_a_venv.reset_calls[-1]["env_idx"] == [1]
    assert task_b_venv.reset_calls[-1]["env_idx"] == [0]


def test_robotwin_multitask_chunk_step_scatters_obs_in_global_order(
    monkeypatch, tmp_path
):
    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "train_seeds.json"
    _write_seed_file(seeds_path)
    env = RoboTwinEnv(
        _make_cfg(seeds_path),
        num_envs=4,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    actions = np.zeros((4, 2, 14), dtype=np.float32)

    obs_list, _rewards, _terminations, _truncations, infos_list = env.chunk_step(
        actions
    )

    assert obs_list[-1]["task_descriptions"] == [
        "task_a-instruction-0",
        "task_b-instruction-0",
        "task_a-instruction-1",
        "task_b-instruction-1",
    ]
    assert infos_list[-1]["task"] == ["task_a", "task_b", "task_a", "task_b"]
    assert env.venvs["task_a"].step_calls[-1].shape == (2, 2, 14)
    assert env.venvs["task_b"].step_calls[-1].shape == (2, 2, 14)

