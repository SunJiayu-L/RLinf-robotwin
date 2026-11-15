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

import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.robotwin.seed_utils import (
    partition_episode_manifest,
    partition_success_seeds,
)
from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboTwinEnv"]


class RoboTwinEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.base_seed = env_seed
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations

        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_custom_reward = cfg.use_custom_reward

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg
        self.record_metrics = record_metrics
        self._is_start = True

        self.task_name = cfg.task_config.task_name
        self.task_names = self._resolve_task_names()
        self.is_multitask = len(self.task_names) > 1
        self._init_task_assignment()

        self.center_crop = cfg.get("center_crop", False)
        self.episode_manifest = None
        self.task_episode_manifests = {}
        self.task_success_seeds = {}
        self._task_current_episode_indices = {}
        self._task_current_seed_indices = {}
        self.current_instructions = None
        # Keep observed text separate: current_instructions controls manifest
        # resets, and filling it from observations would change train sampling.
        self._observed_instructions = [""] * self.num_envs
        self._episode_log_written = np.zeros(self.num_envs, dtype=bool)
        self._episode_ids = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_log_path = None
        self._init_reset_state_ids()
        self._init_episode_log()

        self._init_env()

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        if self.record_metrics:
            self._init_metrics()
            self._elapsed_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def _init_env(self):
        mp.set_start_method("spawn", force=True)
        os.environ["ASSETS_PATH"] = self.cfg.assets_path

        from robotwin.envs.vector_env import VectorEnv

        if self.is_multitask:
            self.venvs = {}
            for task_name, global_indices in self.task_to_global_indices.items():
                task_config = deepcopy(
                    OmegaConf.to_container(self.cfg.task_config, resolve=True)
                )
                task_config["task_name"] = task_name
                env_seeds = [int(self.reset_state_ids[idx]) for idx in global_indices]
                instructions = (
                    [self.current_instructions[idx] for idx in global_indices]
                    if self.current_instructions is not None
                    else None
                )
                self.venvs[task_name] = VectorEnv(
                    task_config=task_config,
                    n_envs=len(global_indices),
                    env_seeds=env_seeds,
                    instructions=instructions,
                    instruction_type=self.cfg.get("instruction_type", "seen"),
                    strict_episode_manifest=self._uses_multitask_manifest,
                )
            return

        env_seeds = self.reset_state_ids.tolist()

        self.venv = VectorEnv(
            task_config=OmegaConf.to_container(self.cfg.task_config, resolve=True),
            n_envs=self.num_envs,
            env_seeds=env_seeds,
            instructions=self.current_instructions,
            instruction_type=self.cfg.get("instruction_type", "seen"),
            strict_episode_manifest=self.episode_manifest is not None,
        )

    def _resolve_task_names(self) -> list[str]:
        task_names = self.cfg.get("task_names", None)
        if task_names is None:
            return [str(self.task_name)]
        task_names = [str(task_name) for task_name in task_names]
        if len(task_names) == 0:
            raise ValueError("RoboTwin task_names must not be empty")
        return task_names

    def _init_task_assignment(self):
        self.env_task_names = [self.task_name] * self.num_envs
        self.env_task_ids = torch.zeros(self.num_envs, dtype=torch.long)
        self.env_local_indices = [0] * self.num_envs
        self.task_to_global_indices = {self.task_name: list(range(self.num_envs))}
        self.task_to_group_indices = {self.task_name: list(range(self.num_group))}
        self._uses_multitask_manifest = False

        if not self.is_multitask:
            return
        distribute_tasks = self.cfg.get("distribute_tasks_across_workers", False)
        available_groups = self.num_group * (
            self.total_num_processes if distribute_tasks else 1
        )
        if len(self.task_names) > available_groups:
            raise ValueError(
                "RoboTwin multi-task mode requires at least one env group per task: "
                f"{len(self.task_names)} tasks > {available_groups} groups"
            )

        self.env_task_names = [""] * self.num_envs
        self.env_task_ids = torch.empty(self.num_envs, dtype=torch.long)
        self.env_local_indices = [0] * self.num_envs
        self.task_to_global_indices = {task_name: [] for task_name in self.task_names}
        self.task_to_group_indices = {task_name: [] for task_name in self.task_names}

        for group_idx in range(self.num_group):
            global_offset = self.seed_offset * self.num_group if distribute_tasks else 0
            task_id = (global_offset + group_idx) % len(self.task_names)
            task_name = self.task_names[task_id]
            self.task_to_group_indices[task_name].append(group_idx)
            for offset in range(self.group_size):
                env_idx = group_idx * self.group_size + offset
                self.env_task_names[env_idx] = task_name
                self.env_task_ids[env_idx] = task_id
                self.env_local_indices[env_idx] = len(
                    self.task_to_global_indices[task_name]
                )
                self.task_to_global_indices[task_name].append(env_idx)

        # A worker may own only a subset of the global task list. Do not create
        # empty VectorEnvs or partition manifests for tasks absent on this rank.
        self.task_to_global_indices = {
            task: indices
            for task, indices in self.task_to_global_indices.items()
            if indices
        }
        self.task_to_group_indices = {
            task: self.task_to_group_indices[task]
            for task in self.task_to_global_indices
        }
        self.task_worker_ranks = {}
        for task in self.task_to_global_indices:
            task_id = self.task_names.index(task)
            self.task_worker_ranks[task] = [
                rank
                for rank in range(self.total_num_processes)
                if any(
                    ((rank * self.num_group if distribute_tasks else 0) + group)
                    % len(self.task_names)
                    == task_id
                    for group in range(self.num_group)
                )
            ]

    def _init_episode_log(self):
        episode_log_path = self.cfg.get("episode_log_path", None)
        if episode_log_path is None:
            return

        path = Path(episode_log_path)
        suffix = path.suffix or ".jsonl"
        self._episode_log_path = path.with_name(
            f"{path.stem}.worker-{self.seed_offset}{suffix}"
        )
        self._episode_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode_log_path.touch(exist_ok=True)
        self._episode_log_session = uuid4().hex

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.first_success_steps = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )

    def _reset_metrics(self, env_idx=None):
        indices = self._normalize_env_indices(env_idx)
        self._episode_log_written[indices] = False
        self._episode_ids[indices] += 1
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                self.first_success_steps[mask] = -1
                self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self.first_success_steps[:] = -1
                self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            if isinstance(infos["success"], list):
                infos["success"] = torch.as_tensor(
                    np.array(infos["success"]).reshape(-1), device=self.device
                )
            newly_successful = infos["success"] & ~self.success_once
            self.first_success_steps[newly_successful] = self.elapsed_steps[
                newly_successful
            ]
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
            episode_info["success_step"] = self.first_success_steps.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["seed"] = self.reset_state_ids.to(self.device).clone()
        if self.is_multitask:
            episode_info["task_id"] = self.env_task_ids.to(self.device).clone()
        infos["episode"] = episode_info
        return infos

    def _write_episode_records(self, dones, infos):
        if self._episode_log_path is None or not dones.any():
            return

        records = []
        pending_indices = [
            idx
            for idx in torch.nonzero(dones, as_tuple=False).flatten().tolist()
            if not self._episode_log_written[idx]
        ]
        for env_idx in pending_indices:
            records.append(
                {
                    "task": self.env_task_names[env_idx]
                    if self.is_multitask
                    else self.task_name,
                    "worker": self.seed_offset,
                    "env_index": env_idx,
                    "seed": int(self.reset_state_ids[env_idx]),
                    "instruction": self._observed_instructions[env_idx],
                    "instruction_source": "observation",
                    "episode_id": int(self._episode_ids[env_idx]),
                    "session_id": self._episode_log_session,
                    "success": bool(self.success_once[env_idx].item()),
                    "success_step": (
                        int(self.first_success_steps[env_idx].item())
                        if self.success_once[env_idx]
                        else None
                    ),
                    "total_steps": int(self.elapsed_steps[env_idx].item()),
                }
            )

        with self._episode_log_path.open("a", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record, ensure_ascii=True) + "\n")
        self._episode_log_written[pending_indices] = True

    def center_and_crop(self, image, center_crop=False):
        image = np.array(image)

        image = Image.fromarray(image).convert("RGB")
        if center_crop:
            image = center_crop_image(image)
        return np.array(image)

    def _extract_obs_image(self, raw_obs):
        batch_images = []
        batch_wrist_images = []
        batch_states = []
        batch_instructions = []
        for env_idx, obs in enumerate(raw_obs):
            instruction = obs["instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(
                    f"Empty RoboTwin instruction for env {env_idx}, "
                    f"seed {int(self.reset_state_ids[env_idx])}"
                )
            self._observed_instructions[env_idx] = instruction
            batch_images.append(
                self.center_and_crop(obs["full_image"], center_crop=self.center_crop)
            )
            wrist_images = []
            if "left_wrist_image" in obs and obs["left_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["left_wrist_image"], center_crop=self.center_crop
                    )
                )
            if "right_wrist_image" in obs and obs["right_wrist_image"] is not None:
                wrist_images.append(
                    self.center_and_crop(
                        obs["right_wrist_image"], center_crop=self.center_crop
                    )
                )
            if len(wrist_images) > 0:
                batch_wrist_images.append(
                    torch.stack([torch.from_numpy(img) for img in wrist_images])
                )
            batch_states.append(obs["state"])
            batch_instructions.append(obs["instruction"])

        batch_images = torch.stack([torch.from_numpy(img) for img in batch_images])
        if len(batch_wrist_images) > 0:
            batch_wrist_images = torch.stack(batch_wrist_images)
        else:
            batch_wrist_images = None
        batch_states = torch.stack([torch.from_numpy(state) for state in batch_states])

        extracted_obs = {
            "main_images": batch_images,
            "wrist_images": batch_wrist_images,
            "states": batch_states,
            "task_descriptions": batch_instructions,
        }

        return extracted_obs

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations

        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _cal_chunk_rewards(self, step_reward, chunk_step, terminations, infos):
        n_steps_to_run = np.array(
            [[0] for i in range(self.num_envs)]
        )  # infos.get("n_steps_to_run", np.array([[0] for i in range(self.num_envs)]))

        n_steps_to_run = torch.as_tensor(
            np.array(n_steps_to_run).reshape(-1), device=self.device
        )
        chunk_rewards = torch.zeros(self.num_envs, chunk_step, device=self.device)
        for env_id in range(self.num_envs):
            steps_left = n_steps_to_run[env_id]
            reward = step_reward[env_id]
            start_idx = chunk_step - steps_left - 1

            if terminations[env_id] and start_idx > 0:
                if self.use_rel_reward:
                    chunk_rewards[env_id, start_idx] = reward
                else:
                    chunk_rewards[env_id, start_idx:] = reward

        return chunk_rewards

    def reset(
        self,
        env_idx: Optional[Union[int, list[int]]] = None,
        env_seeds=None,
    ):
        if self._is_start:
            self._is_start = False

        if self.is_multitask:
            return self._multitask_reset(env_idx=env_idx, env_seeds=env_seeds)

        if env_seeds is None:
            if env_idx is None:
                env_seeds = self.reset_state_ids.tolist()
                instructions = self.current_instructions
            else:
                indices = [env_idx] if isinstance(env_idx, int) else env_idx
                env_seeds = [int(self.reset_state_ids[idx]) for idx in indices]
                instructions = (
                    [self.current_instructions[idx] for idx in indices]
                    if self.current_instructions is not None
                    else None
                )
        else:
            instructions = None

        self.venv.reset(
            env_idx=env_idx,
            env_seeds=env_seeds,
            instructions=instructions,
        )
        raw_obs = self.venv.get_obs()
        infos = {}

        self._reset_metrics(env_idx)

        extracted_obs = self._extract_obs_image(raw_obs)

        return extracted_obs, infos

    def _multitask_reset(self, env_idx=None, env_seeds=None):
        indices = self._normalize_env_indices(env_idx)
        env_seed_by_idx = None
        if env_seeds is not None:
            env_seed_by_idx = {
                idx: int(seed) for idx, seed in zip(indices, env_seeds, strict=True)
            }

        for task_name, task_indices in self._group_indices_by_task(indices).items():
            local_indices = [self.env_local_indices[idx] for idx in task_indices]
            task_env_seeds = [
                env_seed_by_idx[idx]
                if env_seed_by_idx is not None
                else int(self.reset_state_ids[idx])
                for idx in task_indices
            ]
            instructions = (
                [self.current_instructions[idx] for idx in task_indices]
                if self.current_instructions is not None
                else None
            )
            self.venvs[task_name].reset(
                env_idx=local_indices,
                env_seeds=task_env_seeds,
                instructions=instructions,
            )

        raw_obs = self._get_multitask_raw_obs()
        infos = {}
        self._reset_metrics(env_idx)
        return self._extract_obs_image(raw_obs), infos

    def step(
        self, actions: Union[torch.Tensor, np.ndarray, dict] = None, auto_reset=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)

        # [n_envs, horizon, action_dim]
        if len(actions.shape) == 2:
            # [n_envs, action_dim] -> [n_envs, 1, action_dim]
            actions = actions[:, None, :]

        if self.is_multitask:
            raw_obs, step_reward, terminations, truncations, info_list = (
                self._multitask_env_step(actions)
            )
        else:
            raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
                actions
            )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)

        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        self._elapsed_steps += actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        dones = torch.logical_or(terminations, truncations)
        self._write_episode_records(dones, infos)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions, *, record_full_obs: bool = False):
        """Execute one planned chunk, optionally returning each action's observation.

        Full observations are captured inside the native execution loop. Splitting
        the input into separate step calls would change RoboTwin's TOPP planning.
        """
        if record_full_obs and self.is_multitask:
            raise NotImplementedError(
                "Full chunk observations currently require a single-task environment"
            )
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        # chunk_actions: [num_envs, chunk_step, action_dim]
        num_envs = chunk_actions.shape[0]
        chunk_step = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        record_full_obs = record_full_obs or (
            not self.is_multitask
            and self.cfg.get("is_eval", False)
            and self.video_cfg.get("save_video", False)
        )

        if self.is_multitask:
            raw_obs, step_reward, terminations, truncations, info_list = (
                self._multitask_env_step(chunk_actions)
            )
        elif record_full_obs:
            raw_obs_list, step_reward, terminations, truncations, info_list = (
                self.venv.step_with_full_obs(chunk_actions)
            )
            raw_obs = raw_obs_list[-1]
        else:
            raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
                chunk_actions
            )
        if record_full_obs:
            obs_list = [self._extract_obs_image(obs) for obs in raw_obs_list]
        else:
            obs_list.append(self._extract_obs_image(raw_obs))
        infos = list_of_dict_to_dict_of_list(info_list)
        infos_list = [{} for _ in range(len(obs_list) - 1)] + [infos]
        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        chunk_rewards = self._cal_chunk_rewards(
            step_reward, chunk_step, terminations, infos
        )

        self._elapsed_steps += chunk_actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        past_dones = torch.logical_or(terminations, truncations)
        self._write_episode_records(past_dones, infos)
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        chunk_terminations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_terminations[:, -1] = terminations

        chunk_truncations = torch.zeros((num_envs, chunk_step), dtype=bool)
        chunk_truncations[:, -1] = truncations

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        final_info = infos.copy()
        if self.cfg.is_eval:
            if self.episode_manifest is not None and (
                self._current_episode_index + env_idx.numel()
                > len(self.episode_manifest)
            ):
                infos["final_observation"] = final_obs
                infos["final_info"] = final_info
                infos["_final_info"] = dones
                infos["_final_observation"] = dones
                infos["_elapsed_steps"] = dones
                return extracted_obs, infos
            if self._uses_multitask_manifest:
                for task_name, task_done in self._group_indices_by_task(
                    env_idx.tolist()
                ).items():
                    start = self._task_current_episode_indices[task_name]
                    if start + len(task_done) > len(
                        self.task_episode_manifests[task_name]
                    ):
                        infos["final_observation"] = final_obs
                        infos["final_info"] = final_info
                        infos["_final_info"] = dones
                        infos["_final_observation"] = dones
                        infos["_elapsed_steps"] = dones
                        return extracted_obs, infos
            self.update_reset_state_ids(env_idx=env_idx)

        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())
        # Gymnasium calls this final observation, but it is the true next obs.
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return extracted_obs, infos

    def close(self, clear_cache=True):
        if self.is_multitask and hasattr(self, "venvs"):
            for venv in self.venvs.values():
                venv.close(clear_cache)
            return
        if hasattr(self, "venv"):
            self.venv.close(clear_cache)

    def sample_action_space(self):
        return np.random.randn(self.num_envs, self.horizon, 14)

    def begin_evaluation(self):
        """Rewind finite manifests so each PPO validation uses the same episodes."""
        if not self.cfg.get("is_eval", False):
            return
        if self._uses_multitask_manifest:
            self._task_current_episode_indices = dict.fromkeys(
                self.task_to_global_indices, 0
            )
            self._multitask_reset_state_ids_initialized = False
            self.update_reset_state_ids()
            self._multitask_reset_state_ids_initialized = True
        elif self.episode_manifest is not None:
            self._current_episode_index = 0
            self._update_manifest_episodes()

    def _init_reset_state_ids(self):
        if not hasattr(self, "_generator"):
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)

        if self.is_multitask:
            self._init_multitask_reset_state_ids()
            return

        if self.cfg.get("seeds_path", None) is not None and os.path.exists(
            self.cfg.seeds_path
        ):
            with open(self.cfg.seeds_path, "r") as f:
                data = json.load(f)
            task_data = data[self.task_name]
            episodes = task_data.get("episodes", None)
            if episodes is not None:
                self.episode_manifest = partition_episode_manifest(
                    episodes,
                    seed_offset=self.seed_offset,
                    total_num_processes=self.total_num_processes,
                    num_group=self.num_group,
                )
                self._current_episode_index = 0
                self.success_seeds = None
                self._current_seed_index = 0
                self.update_reset_state_ids()
                return

            success_seeds = task_data.get("success_seeds", None)
            if success_seeds is not None:
                success_seeds = torch.as_tensor(success_seeds, dtype=torch.long)
                self.success_seeds = partition_success_seeds(
                    success_seeds,
                    base_seed=self.base_seed,
                    seed_offset=self.seed_offset,
                    total_num_processes=self.total_num_processes,
                    num_group=self.num_group,
                )
                self._current_seed_index = 0
            else:
                self.success_seeds = None
                self._current_seed_index = 0
        else:
            self.success_seeds = None
            self._current_seed_index = 0

        self.update_reset_state_ids()

    def update_reset_state_ids(self, env_idx=None):
        if self.is_multitask:
            self._update_multitask_reset_state_ids(env_idx)
            return

        if self.episode_manifest is not None:
            self._update_manifest_episodes(env_idx)
            return

        if self.use_fixed_reset_state_ids and hasattr(self, "reset_state_ids"):
            return

        if env_idx is not None and hasattr(self, "reset_state_ids"):
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            for idx in env_idx:
                self.reset_state_ids[idx] = reset_state_ids[idx]
        else:
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            self.reset_state_ids = reset_state_ids

    def _update_manifest_episodes(self, env_idx=None):
        if env_idx is None:
            indices = list(range(self.num_envs))
        elif isinstance(env_idx, int):
            indices = [env_idx]
        else:
            indices = [
                int(idx)
                for idx in (
                    env_idx.tolist() if isinstance(env_idx, torch.Tensor) else env_idx
                )
            ]

        end = self._current_episode_index + len(indices)
        if end > len(self.episode_manifest):
            raise RuntimeError(
                "Episode manifest exhausted before evaluation completed: "
                f"requested through {end}, only {len(self.episode_manifest)} available"
            )

        selected = self.episode_manifest[self._current_episode_index : end]
        if not hasattr(self, "reset_state_ids"):
            self.reset_state_ids = torch.empty(self.num_envs, dtype=torch.long)
            self.current_instructions = [""] * self.num_envs

        for idx, episode in zip(indices, selected, strict=True):
            self.reset_state_ids[idx] = int(episode["seed"])
            self.current_instructions[idx] = str(episode["instruction"])

        self._current_episode_index = end

    def check_seeds(self, seeds):
        if self.is_multitask:
            return {
                task_name: venv.check_seeds(seeds)
                for task_name, venv in self.venvs.items()
                if hasattr(venv, "check_seeds")
            }
        results = self.venv.check_seeds(seeds)

        return results

    def _init_multitask_reset_state_ids(self):
        seeds_data = {}
        if self.cfg.get("seeds_path", None) is not None and os.path.exists(
            self.cfg.seeds_path
        ):
            with open(self.cfg.seeds_path, "r") as f:
                seeds_data = json.load(f)

        manifest_tasks = set()
        active_tasks = set(self.task_to_global_indices)
        for task_name in self.task_to_global_indices:
            task_workers = self.task_worker_ranks[task_name]
            task_worker_offset = task_workers.index(self.seed_offset)
            task_data = seeds_data.get(task_name, {})
            episodes = task_data.get("episodes", None)
            if episodes is not None:
                manifest_tasks.add(task_name)
                self.task_episode_manifests[task_name] = partition_episode_manifest(
                    episodes,
                    seed_offset=task_worker_offset,
                    total_num_processes=len(task_workers),
                    num_group=len(self.task_to_group_indices[task_name]),
                )
                self._task_current_episode_indices[task_name] = 0
                continue

            success_seeds = task_data.get("success_seeds", None)
            if success_seeds is not None:
                self.task_success_seeds[task_name] = partition_success_seeds(
                    torch.as_tensor(success_seeds, dtype=torch.long),
                    base_seed=self.base_seed,
                    seed_offset=task_worker_offset,
                    total_num_processes=len(task_workers),
                    num_group=max(1, len(self.task_to_group_indices[task_name])),
                )
            self._task_current_seed_indices[task_name] = 0

        if manifest_tasks and manifest_tasks != active_tasks:
            raise ValueError(
                "RoboTwin multi-task manifest evaluation requires every task to "
                f"provide episodes. Missing: {sorted(active_tasks - manifest_tasks)}"
            )

        self._uses_multitask_manifest = bool(manifest_tasks)
        self.reset_state_ids = torch.empty(self.num_envs, dtype=torch.long)
        self.current_instructions = [""] * self.num_envs if manifest_tasks else None
        self._multitask_reset_state_ids_initialized = False
        self.update_reset_state_ids()
        self._multitask_reset_state_ids_initialized = True

    def _update_multitask_reset_state_ids(self, env_idx=None):
        if self.use_fixed_reset_state_ids and getattr(
            self, "_multitask_reset_state_ids_initialized", False
        ):
            return

        indices = self._normalize_env_indices(env_idx)
        if self._uses_multitask_manifest:
            self._update_multitask_manifest_episodes(indices)
            return

        if env_idx is None:
            for task_name, group_indices in self.task_to_group_indices.items():
                seeds = self._next_task_reset_seeds(task_name, len(group_indices))
                for group_idx, seed in zip(group_indices, seeds, strict=True):
                    start = group_idx * self.group_size
                    end = start + self.group_size
                    self.reset_state_ids[start:end] = seed
            return

        for task_name, task_indices in self._group_indices_by_task(indices).items():
            seeds = self._next_task_reset_seeds(task_name, len(task_indices))
            for idx, seed in zip(task_indices, seeds, strict=True):
                self.reset_state_ids[idx] = seed

    def _update_multitask_manifest_episodes(self, indices: list[int]):
        for task_name, task_indices in self._group_indices_by_task(indices).items():
            start = self._task_current_episode_indices[task_name]
            end = start + len(task_indices)
            manifest = self.task_episode_manifests[task_name]
            if end > len(manifest):
                raise RuntimeError(
                    "Episode manifest exhausted before evaluation completed for "
                    f"task {task_name}: requested through {end}, only {len(manifest)} "
                    "available"
                )

            for idx, episode in zip(task_indices, manifest[start:end], strict=True):
                self.reset_state_ids[idx] = int(episode["seed"])
                self.current_instructions[idx] = str(episode["instruction"])
            self._task_current_episode_indices[task_name] = end

    def _next_task_reset_seeds(self, task_name: str, count: int) -> torch.Tensor:
        success_seeds = self.task_success_seeds.get(task_name, None)
        if success_seeds is None or success_seeds.numel() == 0:
            return torch.randint(
                low=10000,
                high=200000,
                size=(count,),
                generator=self._generator,
            )

        current_index = self._task_current_seed_indices.get(task_name, 0)
        indices = (
            torch.arange(count, device=success_seeds.device) + current_index
        ) % success_seeds.numel()
        self._task_current_seed_indices[task_name] = (
            current_index + count
        ) % success_seeds.numel()
        return success_seeds[indices]

    def _normalize_env_indices(self, env_idx=None) -> list[int]:
        if env_idx is None:
            return list(range(self.num_envs))
        if isinstance(env_idx, int):
            return [env_idx]
        if isinstance(env_idx, torch.Tensor):
            return [int(idx) for idx in env_idx.flatten().tolist()]
        return [int(idx) for idx in env_idx]

    def _group_indices_by_task(self, indices: list[int]) -> dict[str, list[int]]:
        grouped = defaultdict(list)
        for idx in indices:
            grouped[self.env_task_names[idx]].append(idx)
        return dict(grouped)

    def _get_multitask_raw_obs(self) -> list[dict]:
        raw_obs = [None] * self.num_envs
        for task_name, global_indices in self.task_to_global_indices.items():
            task_raw_obs = self.venvs[task_name].get_obs()
            for local_idx, global_idx in enumerate(global_indices):
                raw_obs[global_idx] = task_raw_obs[local_idx]
        return raw_obs

    def _multitask_env_step(self, actions):
        raw_obs = [None] * self.num_envs
        rewards = [None] * self.num_envs
        terminations = [None] * self.num_envs
        truncations = [None] * self.num_envs
        infos = [None] * self.num_envs

        for task_name, global_indices in self.task_to_global_indices.items():
            task_actions = actions[global_indices]
            (
                task_raw_obs,
                task_rewards,
                task_terminations,
                task_truncations,
                task_infos,
            ) = self.venvs[task_name].step(task_actions)
            for local_idx, global_idx in enumerate(global_indices):
                raw_obs[global_idx] = task_raw_obs[local_idx]
                rewards[global_idx] = task_rewards[local_idx]
                terminations[global_idx] = task_terminations[local_idx]
                truncations[global_idx] = task_truncations[local_idx]
                infos[global_idx] = task_infos[local_idx]

        return raw_obs, rewards, terminations, truncations, infos
