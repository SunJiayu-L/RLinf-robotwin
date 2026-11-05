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
import pytest
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
                "instructions": list(instructions)
                if instructions is not None
                else None,
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


SIX_TASKS = [
    "beat_block_hammer",
    "place_phone_stand",
    "pick_dual_bottles",
    "lift_pot",
    "move_can_pot",
    "place_a2b_left",
]


def test_robotwin_six_tasks_dispatch_actions_and_cycle_seeds(monkeypatch, tmp_path):
    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "six_train.json"
    pools = {
        task: {"success_seeds": list(range(1000 * (i + 1), 1000 * (i + 1) + 16))}
        for i, task in enumerate(SIX_TASKS)
    }
    seeds_path.write_text(json.dumps(pools))
    cfg = _make_cfg(seeds_path)
    cfg.task_names = SIX_TASKS
    cfg.task_config.task_name = SIX_TASKS[0]
    env = RoboTwinEnv(cfg, 6, 3, 8, None)
    actions = np.broadcast_to(np.arange(6)[:, None, None], (6, 2, 14)).copy()
    obs, rewards, _, _, infos = env.chunk_step(actions)
    assert [
        name.split("-instruction-")[0] for name in obs[-1]["task_descriptions"]
    ] == SIX_TASKS
    assert infos[-1]["task"] == SIX_TASKS
    assert rewards.shape == (6, 2)
    for i, task in enumerate(SIX_TASKS):
        assert np.all(env.venvs[task].step_calls[-1] == i)
    for _ in range(8):
        env.update_reset_state_ids()
        for i, task in enumerate(SIX_TASKS):
            assert int(env.reset_state_ids[i]) in pools[task]["success_seeds"]


def test_distributed_six_task_manifests_cover_96_unique_episodes(monkeypatch, tmp_path):
    _install_fake_robotwin(monkeypatch)
    path = tmp_path / "distributed_eval.json"
    path.write_text(
        json.dumps(
            {
                task: {
                    "episodes": [
                        {"seed": 1000 * (i + 1) + j, "instruction": f"{task}:{j}"}
                        for j in range(96)
                    ]
                }
                for i, task in enumerate(SIX_TASKS)
            }
        )
    )
    covered = {task: [] for task in SIX_TASKS}
    counts = dict.fromkeys(SIX_TASKS, 0)
    for rank in range(8):
        cfg = _make_cfg(path)
        cfg.task_names = SIX_TASKS
        cfg.task_config.task_name = SIX_TASKS[0]
        cfg.is_eval = True
        cfg.distribute_tasks_across_workers = True
        env = RoboTwinEnv(cfg, 3, rank, 8, None)
        expected = [SIX_TASKS[(rank * 3 + slot) % 6] for slot in range(3)]
        assert env.env_task_names == expected
        assert set(env.venvs) == set(expected)
        for task in expected:
            counts[task] += 1
            episodes = env.task_episode_manifests[task]
            assert len(episodes) == 24
            covered[task].extend(episode["seed"] for episode in episodes)
        first_seeds = env.reset_state_ids.clone()
        for _ in range(23):
            env.update_reset_state_ids()
        env.begin_evaluation()
        assert (env.reset_state_ids == first_seeds).all()
        actions = np.broadcast_to(np.arange(3)[:, None, None], (3, 2, 14)).copy()
        obs, _, _, _, _ = env.chunk_step(actions)
        assert [text.split(":")[0] for text in obs[-1]["task_descriptions"]] == expected
        for local, task in enumerate(expected):
            assert np.all(env.venvs[task].step_calls[-1] == local)
    assert all(count == 4 for count in counts.values())
    assert all(len(seeds) == len(set(seeds)) == 96 for seeds in covered.values())


@pytest.mark.parametrize("multitask", [False, True])
def test_robotwin_eval_manifest_rewinds_between_validations(
    monkeypatch, tmp_path, multitask
):
    _install_fake_robotwin(monkeypatch)
    tasks = SIX_TASKS if multitask else SIX_TASKS[:1]
    seeds_path = tmp_path / "eval_manifest.json"
    seeds_path.write_text(
        json.dumps(
            {
                task: {
                    "episodes": [
                        {
                            "seed": 1000 * (i + 1) + j,
                            "instruction": f"{task} episode {j}",
                        }
                        for j in range(16)
                    ]
                }
                for i, task in enumerate(tasks)
            }
        )
    )
    cfg = _make_cfg(seeds_path)
    cfg.task_names = tasks
    cfg.task_config.task_name = tasks[0]
    cfg.is_eval = True
    env = RoboTwinEnv(cfg, len(tasks), 3, 8, None)
    initial_seeds = env.reset_state_ids.clone()
    initial_instructions = list(env.current_instructions)
    env.update_reset_state_ids()
    assert not (env.reset_state_ids == initial_seeds).all()
    # Both validations must begin at the same ordered seed/instruction pair.
    for _ in range(2):
        env.begin_evaluation()
        assert (env.reset_state_ids == initial_seeds).all()
        assert env.current_instructions == initial_instructions
        env.update_reset_state_ids()


@pytest.mark.parametrize("single_task", [False, True])
def test_episode_log_uses_observed_instruction_without_changing_reset_policy(
    monkeypatch, tmp_path, single_task
):
    import torch

    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "seeds.json"
    _write_seed_file(seeds_path)
    cfg = _make_cfg(seeds_path)
    cfg.episode_log_path = str(tmp_path / "train.jsonl")
    if single_task:
        cfg.task_names = ["task_a"]
    env = RoboTwinEnv(
        cfg, num_envs=4, seed_offset=0, total_num_processes=1, worker_info=None
    )
    observations, _ = env.reset()
    assert env.current_instructions is None
    env._elapsed_steps[:] = 5
    env.success_once[:] = True
    env.first_success_steps[:] = 5
    env._write_episode_records(torch.ones(4, dtype=torch.bool), {})
    rows = [json.loads(line) for line in env._episode_log_path.read_text().splitlines()]
    assert len(rows) == 4
    assert [row["instruction"] for row in rows] == observations["task_descriptions"]
    assert all(row["instruction"] for row in rows)
    assert all(row["instruction_source"] == "observation" for row in rows)
    assert env.current_instructions is None
    env.reset()
    vectors = [env.venv] if single_task else list(env.venvs.values())
    assert all(vector.reset_calls[-1]["instructions"] is None for vector in vectors)


def test_episode_log_records_only_first_terminal_and_rearms_on_partial_reset(
    monkeypatch, tmp_path
):
    import torch

    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "seeds.json"
    _write_seed_file(seeds_path)
    cfg = _make_cfg(seeds_path)
    cfg.episode_log_path = str(tmp_path / "train.jsonl")
    env = RoboTwinEnv(
        cfg, num_envs=4, seed_offset=0, total_num_processes=1, worker_info=None
    )
    env.reset()
    done = torch.tensor([True, False, False, False])
    env._elapsed_steps[0] = 5
    env._write_episode_records(done, {})
    env._elapsed_steps[0] = 10
    env._write_episode_records(done, {})
    rows = [json.loads(line) for line in env._episode_log_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["total_steps"] == 5
    first_episode_id = rows[0]["episode_id"]
    env.reset(env_idx=[1])
    env._write_episode_records(done, {})
    assert len(env._episode_log_path.read_text().splitlines()) == 1
    env.reset(env_idx=[0])
    env._elapsed_steps[0] = 3
    env._write_episode_records(done, {})
    rows = [json.loads(line) for line in env._episode_log_path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[-1]["episode_id"] == first_episode_id + 1


def test_logged_instruction_tracks_actual_observation_changes(monkeypatch, tmp_path):
    import torch

    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "seeds.json"
    _write_seed_file(seeds_path)
    cfg = _make_cfg(seeds_path)
    cfg.task_names = ["task_a"]
    cfg.episode_log_path = str(tmp_path / "train.jsonl")
    env = RoboTwinEnv(
        cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None
    )
    env.reset()
    env.venv.instructions[0] = "actual instruction returned by simulator"
    model_input = env._extract_obs_image(env.venv.get_obs())
    env._elapsed_steps[:] = 4
    env._write_episode_records(torch.tensor([True, False]), {})
    row = json.loads(env._episode_log_path.read_text())
    assert row["instruction"] == model_input["task_descriptions"][0]
    assert row["instruction"] == "actual instruction returned by simulator"


def test_empty_simulator_instruction_fails_before_model_input(monkeypatch, tmp_path):
    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "seeds.json"
    _write_seed_file(seeds_path)
    cfg = _make_cfg(seeds_path)
    cfg.task_names = ["task_a"]
    env = RoboTwinEnv(
        cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None
    )
    env.reset()
    raw = env.venv.get_obs()
    raw[0]["instruction"] = " "
    with pytest.raises(ValueError, match="Empty RoboTwin instruction"):
        env._extract_obs_image(raw)


def test_nonchunk_step_also_logs_once(monkeypatch, tmp_path):
    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "seeds.json"
    _write_seed_file(seeds_path)
    cfg = _make_cfg(seeds_path)
    cfg.task_names = ["task_a"]
    cfg.episode_log_path = str(tmp_path / "step.jsonl")
    cfg.max_episode_steps = 1
    env = RoboTwinEnv(
        cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None
    )
    env.reset()
    env.step(np.zeros((2, 1, 14)))
    env.step(np.zeros((2, 1, 14)))
    rows = [json.loads(line) for line in env._episode_log_path.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["total_steps"] == 1 for row in rows)


def test_episode_log_preserves_prior_run_and_distinguishes_sessions(
    monkeypatch, tmp_path
):
    import torch

    _install_fake_robotwin(monkeypatch)
    seeds_path = tmp_path / "seeds.json"
    _write_seed_file(seeds_path)
    cfg = _make_cfg(seeds_path)
    cfg.task_names = ["task_a"]
    cfg.episode_log_path = str(tmp_path / "train.jsonl")
    sessions = []
    for _ in range(2):
        env = RoboTwinEnv(
            cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None
        )
        env.reset()
        env._elapsed_steps[:] = 4
        env._write_episode_records(torch.tensor([True, False]), {})
        sessions.append(env._episode_log_session)
    assert sessions[0] != sessions[1]
    rows = [json.loads(line) for line in env._episode_log_path.read_text().splitlines()]
    assert len(rows) == 2
    assert [row["session_id"] for row in rows] == sessions
